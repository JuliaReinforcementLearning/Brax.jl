module Brax

export BraxEnv

using ReinforcementLearningBase
using CommonRLSpaces
using PyCall
using Random

const brax_envs = PyNULL()
const brax_io_html = PyNULL()
const brax_io_image = PyNULL()
const jax = PyNULL()
const dl = PyNULL()
const jp = PyNULL()

#####

using DLPack

# watch https://github.com/pabloferz/DLPack.jl/pull/27
DLPack.is_col_major(manager::DLPack.DLManagedTensor, val::Val{0}) = true

from_jax(o) = DLPack.wrap(o, o -> @pycall dl.to_dlpack(o)::PyObject)
to_jax(o) = DLPack.share(o, dl.from_dlpack)

#####

mutable struct BraxEnv{B<:Union{Nothing,Int}} <: AbstractEnv
    env::PyObject
    batch_size::B
    key::PyObject
    state::PyObject

    function BraxEnv(name::String; seed=Random.make_seed() |> first, batch_size=nothing, kw...)
        pyenv = brax_envs.create(name; batch_size=batch_size, kw...)
        key = jax.random.PRNGKey(seed)
        env = new{typeof(batch_size)}(pyenv, batch_size, key)
        reset!(env)
        env
    end
end

Base.show(io::IO, ::MIME"text/html", env::BraxEnv{Nothing}) = print(io, brax_io_html.render(env.env.sys, [env.state.qp]))
Base.show(io::IO, ::MIME"image/png", env::BraxEnv{Nothing}) = print(io, brax_io_image.render(env.env.sys, [env.state.qp], width=320, height=240))

Random.seed!(env::BraxEnv, seed) = env.key = jax.random.PRNGKey(seed)

(env::BraxEnv)(action) = env(to_jax(action))
(env::BraxEnv)(action::PyObject) = env.state = jax.jit(env.env.step)(env.state, action)

RLBase.action_space(env::BraxEnv{Nothing}) = Space(Float32, env.env.action_size)
RLBase.action_space(env::BraxEnv) = Space(Float32, env.env.action_size, env.env.batch_size)

RLBase.state_space(env::BraxEnv{Nothing}) = Space(Float32, env.env.observation_size)
RLBase.state_space(env::BraxEnv) = Space(Float32, env.env.observation_size, env.env.batch_size)


function RLBase.reset!(env::BraxEnv)
    key1, key2 = jp.random_split(env.key)
    env.key = key1
    env.state = jax.jit(env.env.reset)(key2)
end

RLBase.state(env::BraxEnv) = env.state.obs |> from_jax
RLBase.reward(env::BraxEnv) = env.state.reward |> from_jax
RLBase.is_terminated(env::BraxEnv) = env.state.done |> from_jax
RLBase.is_terminated(env::BraxEnv{Nothing}) = from_jax(env.state.done)[] |> Bool  # TODO: allow scalar

function __init__()
    copy!(brax_envs, pyimport("brax.envs"))
    copy!(brax_io_html, pyimport("brax.io.html"))
    copy!(brax_io_image, pyimport("brax.io.image"))
    copy!(jax, pyimport("jax"))
    copy!(dl, pyimport("jax.dlpack"))
    copy!(jp, pyimport("brax.jumpy"))

    default_backend = jax.default_backend()
    @info "jax is using $default_backend as the default backend"
end

end # module
