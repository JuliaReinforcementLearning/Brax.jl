module Brax

export BraxEnv

using ReinforcementLearningBase
using CommonRLSpaces
using PyCall
using Random

const brax_envs = PyNULL()
const jax = PyNULL()
const dl = PyNULL()

#####

using DLPack

# watch https://github.com/pabloferz/DLPack.jl/pull/27
DLPack.is_col_major(manager::DLPack.DLManagedTensor, val::Val{0}) = true

from_jax(o) = @pycall dl.to_dlpack(o)::PyObject
to_jax(o) = DLPack.share(o, dl.from_dlpack)

#####

mutable struct BraxEnv{B<:Union{Nothing,Int}} <: AbstractEnv
    env::PyObject
    batch_size::B
    key::PyObject
    state::PyObject

    function BraxEnv(name::String; seed=0, batch_size=nothing, kw...)
        pyenv = brax_envs.create(name; batch_size=batch_size, kw...)
        key = jax.random.PRNGKey(seed)
        env = new{typeof(batch_size)}(pyenv, batch_size, key)
        reset!(env)
        env
    end
end

Random.seed!(env::BraxEnv, seed) = env.key = jax.random.PRNGKey(seed)

(env::BraxEnv)(action) = env(to_jax(action))
(env::BraxEnv)(action::PyObject) = env.state = jax.jit(env.env.step)(env.state, action)

RLBase.action_space(env::BraxEnv{Nothing}) = Space(Float32, env.env.action_size)
RLBase.action_space(env::BraxEnv) = Space(Float32, env.env.action_size, env.env.batch_size)

RLBase.state_space(env::BraxEnv{Nothing}) = Space(Float32, env.env.observation_size)
RLBase.state_space(env::BraxEnv) = Space(Float32, env.env.observation_size, env.env.batch_size)


RLBase.reset!(env::BraxEnv) = env.state = jax.jit(env.env.reset)(env.key)
RLBase.state(env::BraxEnv) = env.state.obs |> from_jax
RLBase.reward(env::BraxEnv) = env.state.reward |> from_jax
RLBase.is_terminated(env::BraxEnv) = env.state.done |> from_jax
RLBase.is_terminated(env::BraxEnv{Nothing}) = from_jax(env.state.done)[]  # TODO: allow scalar

function __init__()
    copy!(brax_envs, pyimport("brax.envs"))
    copy!(jax, pyimport("jax"))
    copy!(dl, pyimport("jax.dlpack"))
end

end # module
