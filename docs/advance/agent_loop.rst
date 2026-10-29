Agent Loop
==========

Last updated: 10/28/2026.

.. versionadded:: 0.4.2
   [status: alpha]

.. warning::
   Agent Loop is ready for use, but the API may change in future releaes.

Agent Loop is designed as general interface for multi-turn rollout and agentic reinforcement learning.

**Design goal**:

- Plugable user defined agent loop
- Provide standard request generate api with different inference frameworks
- Provide request level load balance between multiple inference servers

**Non-goal**:

- How tool is defined and how to call tool

In high level overview, agent loop is given a prompt, run user defined loop: call LLM generate api, call tools, ...
and return the final output. The final output is then calculated reward and used as trajectory for RL training.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_overview.svg?raw=true


API Design
----------

``AgentLoopBase`` class is the abstraction of agent loop, and ``run`` method is the only interface that user need to implement.
The run method, given prompt messages in format: [{"role": "user"}, {"content": "..."}], and additional sampling params,
could do whatever user wants, such as

- call LLM generate api
- call tools: web search, database query, code sandbox, ...
- environment interaction
- reflection
- ...

.. code:: python

   class AgentLoopBase(ABC):
       @abstractmethod
       async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
           """Run agent loop to interact with LLM server and environment.

           Args:
               sampling_params (Dict[str, Any]): LLM sampling params.
               **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

           Returns:
               AgentLoopOutput: Agent loop output.
           """
           raise NotImplementedError

After running user defined loop, run method should return ``AgentLoopOutput``, including prompt token ids,
response token ids, and response mask.

.. code:: python

   class AgentLoopOutput(BaseModel):
       """Agent loop output."""

       prompt_ids: list[int]
       """Prompt token ids."""
       response_ids: list[int]
       """Response token ids including LLM generated token, tool response token."""
       response_mask: list[int]
       """Response mask, 1 for LLM generated token, 0 for tool response token."""

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_output.svg?raw=true

.. note:: AgentLoopOutput only output one trajectory for a given prompt, multiple trajectories output is still under discussion.

Architecture Design
-------------------

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/agent_loop_architecture.png?raw=true

A single PPO step contain two phase: rollout and train. In rollout phase:

1. PPOTrainer sample a batch from dataset and call ``AgentLoopManager.generate_sequences``.
2. AgentLoopManager ``wake_up`` all async LLM server instances, which will sync weights between inference engine(vLLM/SGLang) and training engine(FSDP/Megatron-LM).
3. AgentLoopManager split batch into chunks and send each chunk to ``AgentLoopWorker``.
4. AgentLoopWorker receive chunk and for each prompt, spawn a user defined ``AgentLoopBase`` instance, run ``run`` coroutine until end and get ``AgentLoopOutput``.

.. tip::
   AgentLoopWorker schedules multiple coroutines concurrently. If number of AgentLoopWorker equals batch_size, then each worker is response for one prompt.

In agent loop, when user need LLM generate response:

5. Call ``LLMServerClient.generate`` with prompt_ids.
6. LLMServerClient select a server instance with least request in first turn and send request to it. (In following turns, the request will be sent to the same server instance).
7. AsyncLLMServer receive a request, issue ipc/rpc with model_runner, and generate response. (There's slight differences between vLLM and SGLang, see below).

When all prompts in all AgentLoopWorker finish, AgentLoopManager gather results and return to PPOTrainer.

8. AgentLoopManager ``sleep`` all server instances, which will free kv cache and offload weights to CPU memory.

AsyncLLMServer
~~~~~~~~~~~~~~

AsyncLLMServer is the abstraction of LLM server with two types of generation api:

- `OpenAI chat completion <https://platform.openai.com/docs/api-reference/chat>`_: generate response for the given chat conversation.
- Token in token out: generate response ids for the given token ids.

We have officially supported vLLM and SGLang AsyncLLMServer, both of them implement the two api and are well tested.
Other inference engine should be easy to plug-in by implement the ``AsyncServerBase`` class.

.. code:: python

   class AsyncServerBase(ABC):
       @abstractmethod
       async def chat_completion(self, raw_request: Request) -> JSONResponse:
           """OpenAI chat completion API.

           Args:
               raw_request (Request): raw json request
           
           Returns:
               JSONResponse: json response

           API reference: https://platform.openai.com/docs/api-reference/chat/create
           """
           raise NotImplementedError

       @abstractmethod
       async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
           """Generate response ids given prompt ids.

           Args:
               prompt_ids (List[int]): prompt ids
               sampling_params (Dict[str, Any]): sampling params
               request_id (str): request id

           Returns:
               List[int]: response ids
           """
           raise NotImplementedError


Chat completion vs Token in token out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   The following conclusion is based on our recent experience and is still open to investigation and discussion.

Almost all agent frameworks (LangGraph, CrewAI, LlamaIndex, etc) call LLM with OpenAI chat completion api, and 
keep chat history as messages. So user may expect that we should use the chat completion api in multi-turn rollout.

But based on our recent experience on single-turn training on DAPO and multi-turn training on `retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_,
we found the token_ids from apply the final messages may not equal to the token_ids by concat prompt_ids and response_ids in each turn.

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/multi_turn.png?raw=true

**Where does this inconsistency happened?**

First, the tool parser may alter the content. For example

.. code:: json

   {"role": "assistant", "content": "Let me call a <tool_call>...</tool_call> and get the result"}

After tool_calls extraction, the messages is like this:

.. code:: json

   {"role": "assistant", "content": "Let me call a and get the result", "tool_calls": [{"name": "foo", "arguments": "{}"}]}

Encode the extracted message back is not equal to the original LLM generated response_ids.

Second,  the `decode-encode` may also lead to inconsistency: `Agent-R1 issue#30 <https://github.com/0russwest0/Agent-R1/issues/30#issuecomment-2826155367>`_.

**What is the impact of this inconsistency?**

This inconsistency is not a big problem for serving/agent system, but is critical to RL training.
It causes the trajectory deviate from the policy model distribution. We have observed that apply_chat_template
to the final chat history messages make PPO training not even converged in single-turn.

vLLM
^^^^

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/async_vllm.png?raw=true

For vLLM, the Async LLM Engine is running in same process as the server, and ModelRunner is running in same process as FSDP/Megatron-LM workers.
Async LLM Engine communicate with ModelRunner through ZeroMQ. When server receive a request, it directly call engine to generate response_ids.

SGLang
^^^^^^

.. image:: https://github.com/eric-haibin-lin/verl-community/blob/main/docs/async_sglang.png?raw=true

For SGLang, the Async LLM Engine is running in same process as FSDP/Megatron-LM worker-0, and it spawn multiple subprocesses as ModelRunner.
Also, Async LLM Engine communicate with ModelRunner through ZeroMQ. When server receive a request, it remote call the worker-0 and get response_ids.

LLMServerClient
~~~~~~~~~~~~~~~~~~~~~

LLMServerClient serve as proxy to multiple AsyncLLMServer instances, provides:

- load balance: select a server instance with least request in first turn and send request to it.
- sticky session: bind request_id to server instance, so that the same request_id will be sent to the same server instance in following turns.

LLMServerClient is passed to ``AgentLoopBase.__init__``, whenever user want to interact with LLM in agent loop,
they can call ``LLMServerClient.generate`` to generate response_ids.

.. code:: python

   class LLMServerClient:
       async def generate(
           self,
           request_id,
           *,
           prompt_ids: list[int],
           sampling_params: dict[str, Any],
       ) -> list[int]:
           """Generate tokens from prompt ids.

           Args:
               request_id (str): request id for sticky session.
               prompt_ids (List[int]): List of prompt token ids.
               sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

           Returns:
               List[int]: List of generated token ids.
           """
           ...

Turn-level rewards and episode done
-----------------------------------

.. versionadded:: 0.10.0

A ``BaseTool.execute`` already returns a per-call step reward, and ``ToolAgentLoop`` collects every
step reward of a trajectory in ``extra_fields["tool_rewards"]``. By default that signal never reaches
training: the trajectory's only reward is the episode score, placed on the last response token.
Three opt-in pieces turn the step rewards into per-turn credit, the shape an environment loop needs
(an observation per turn, an action per turn, a reward per cycle, an episode that ends when the
environment says so):

1. **A tool can end the episode.** ``ToolResponse(text=..., done=True)`` makes the loop append the
   response as the final observation and terminate instead of generating another turn
   (``extra_fields["episode_done"]`` records it). The response budget, ``max_assistant_turns`` and
   ``max_user_turns`` still apply; note that the last allowed assistant turn never executes its tool
   calls, so an environment of ``K`` steps needs ``max_assistant_turns >= K + 1`` for the final step
   reward and ``done`` to land. A wall-clock deadline per episode is a planned follow-up.

2. **Every generated turn is located.** With the flag below on, ``extra_fields["generated_turn_spans"]``
   holds the ``[start, end)`` offsets of each assistant turn within ``response_ids`` (the names of
   draft PR #8305, so the two can stack), ``extra_fields["turn_rewards"]`` the step rewards of the
   tool calls each turn made, summed per turn and index-aligned with the spans, and
   ``extra_fields["episode_done"]`` the ``done`` signal; a ``custom_reward_function`` sees all three
   in ``extra_info`` under every reward manager when the flag is on (``naive``, ``dapo``, ``gdpo``,
   ``rate_limited``, ``remote`` then build it with ``RewardManagerBase.extra_info_for_reward_fn``: a copy
   of the dataset's ``extra_info`` merged with copies of the loop's metadata plus ``num_turns``, the
   loop's fields overriding dataset keys of the same name; a dataset whose ``extra_info`` is not a
   mapping is handed over unchanged, with one warning). With the flag off the trajectory metadata and
   each manager's callback contract are exactly what they are today: ``naive`` / ``remote`` merge the
   loop's ``tool_extra_fields`` plus ``num_turns`` as before, ``rate_limited`` merges ``tool_extra_fields``
   as before, ``dapo`` / ``gdpo`` hand the dataset's ``extra_info`` over untouched.

3. **A dense reward tensor** (``actor_rollout_ref.rollout.multi_turn.turn_level_rewards: True``):
   each turn's reward is written on the *last generated token* of its span and the episode score on
   the last generated token of the response, never on a tool-response token. GAE and REINFORCE++
   carry returns across ``response_mask == 0`` spans but drop a reward that sits on one (issue
   #8367), so this placement is what makes them credit turns. ``grpo``, ``rloo`` and
   ``reinforce_plus_plus_baseline`` sum the row and keep behaving episodically. With the flag off
   nothing changes: one score on the last token, as before. Two consequences to keep in mind:
   the row is written by the agent loop, so the reward must be computed there (a rule-based reward,
   or a reward model with ``reward.reward_model.enable_resource_pool=True``; a colocated reward model
   would overwrite the row, and the config validation rejects that combination), and every metric
   that sums ``rm_scores`` (``critic/score/*``, ``val-core/*/reward``, DAPO's group filter) then reports
   the undiscounted return, step rewards plus outcome, instead of the outcome alone. The flag applies
   to validation rollouts too.

**The response-budget boundary.** When the tokens of a tool response (the context merge) would make
the response reach ``response_length``, the loop terminates *before* applying that merge, exactly as it
did before this feature. The turn's step reward and ``done`` were recorded when the tool ran, so
``turn_rewards`` / ``rm_scores`` and ``episode_done`` keep them, but the tool response itself is not part
of ``response_ids``: a reward function that decodes the response never sees that terminal observation,
and ``episode_done`` can be ``True`` while the trajectory ends with the assistant turn that triggered it.
Budget ``response_length`` for the final observation when the scorer needs it.

A minimal device tool, with the configuration that trains on its step rewards:

.. code:: python

   from verl.tools.base_tool import BaseTool
   from verl.tools.schemas import ToolResponse

   class WorkcellTool(BaseTool):
       async def execute(self, instance_id, parameters, **kwargs):
           observation, step_reward, placed = await self.device.step(parameters)  # one cycle
           return ToolResponse(text=observation, done=placed), step_reward, {}

.. code:: yaml

   actor_rollout_ref:
     rollout:
       multi_turn:
         enable: True
         tool_config_path: workcell_tools.yaml
         turn_level_rewards: True
   algorithm:
     adv_estimator: reinforce_plus_plus   # or gae; both read the dense rewards
     gamma: 1.0

Next
----

- :doc:`Agentic RL Training<../start/agentic_rl>`: Quick start agentic RL training with gsm8k dataset.
- `LangGraph MathExpression <https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent/example>`_: Demonstrate how to use LangGraph to build agent loop.
- `Retool <https://github.com/verl-project/verl-recipe/tree/main/retool>`_: End-to-end retool paper reproduction using tool agent.
