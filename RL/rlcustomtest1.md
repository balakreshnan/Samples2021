# Azure Machine learning Deep reinforcement learning

## Using Open Gym and Tensorflow, Keras

## Use Case

- Use reinforcement learning and Train a deep RL model
- Use Open Gym and Tensorflow
- Use Keras
- Install dependencies
- Original Tutorial replicated from - https://github.com/nicknochnack/TensorflowKeras-ReinforcementLearning/blob/master/Deep%20Reinforcement%20Learning.ipynb
- This tutorial is show the above works in Azure Machine learning service.

## Install dependencies

```
!pip install --upgrade tensorflow
!pip install gym
!pip install keras
!pip install keras-rl2
```

```
!sudo apt-get update
!sudo apt-get install -y xvfb ffmpeg
!pip install 'imageio==2.4.0'
!pip install pyvirtualdisplay
!sudo apt-get install -y python-opengl xvfb
```

- When i tested tensorflow version was 2.6.0
- Virtual display is necessary for open gym to render the images or graphics
- Idea is to show we can run all open source code in jupyter lab within Azure Machine learning

## Code

- Display tensorflow version

```
import tensorflow as tf; 
print(tf.__version__)
```

- now imports

```
import gym 
import random
```

```
import os
import io
import base64
from IPython.display import display, HTML
```

- configure the display

```
from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 600))
display.start()
```

```
import matplotlib.pyplot as plt
%matplotlib inline
from IPython import display
```

- Now set the open gym environment

```
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n
```

```
actions
```

- Test if render is working

```
import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()
```

- Create episodes

```
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
```

- Now time to build model

```
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
```

- Define the model

```
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model
```

- Build model

```
model = build_model(states, actions)
```

- Display model

```
model.summary()
```

- Now Deep reinforcement learning includes

```
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
```

- Define the agent

```
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
```

- Build and run model

```
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
```

- Test

```
scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))
```

- test

```
_ = dqn.test(env, nb_episodes=15, visualize=True)
```

- Save model for scoring

```
dqn.save_weights('dqn_weights.h5f', overwrite=True)
```

- delete everything

```
del model
del dqn
del env
```

- Now load the model
- Inferencing testing with new data set
- Setup the environment and load the saved model

```
env = gym.make('CartPole-v0')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
```

- load the weights

```
dqn.load_weights('dqn_weights.h5f')
```

- Run the scoring

```
_ = dqn.test(env, nb_episodes=5, visualize=True)
```