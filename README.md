## Skychain
The simulation of paper *[SkyChain: A Deep Reinforcement Learning-Empowered Dynamic Blockchain Sharding System](https://dl.acm.org/doi/abs/10.1145/3404397.3404460?casa_token=cORVUGinc7MAAAAA%3ANNmTSDChQXOvPDEf6F72O83Eb8lBqlF3Od8S2_Ve66foL0Vb3_wGxEE1aIWaOoUNi1g-DvmlgArbeA)*.

## Installation

1. install [openai gym](https://github.com/openai/gym)

2. install [openai baselines](https://github.com/openai/baselines)

## Register environment

1. Create a new folder ```myEnvDir``` in ```$gym_path/gym/envs```

2. Copy ```./skychainEnv/*``` to ```$myEnvDir```

3. Register the environment in ```$gym_path/gym/envs/__init__.py```, adding the following codes:

```
register(
    id="skychain-env",
    entry_point="gym.envs.myEnvDir.myEnv:myEnvClass",
    max_episode_steps=1000,
)
```

## Test

Now, you can test the codes by running
```
python -m baselines.run --alg=ddpg --env=skychain-env --num_timesteps=1e6 --num_layers=2  --num_hidden=300 --nsteps=100  --layer_norm=True
``` 