# Maximum Entropy Deep Inverse Reinforcement Learning
The purpose of this repository is to get a taste of ```Inverse Reinforcement Learning```. To replicate an expert's behavior in reward-free environments, a reward function should be recreated. Hence, for the implemented "Gridworld" and "ObjectWorld" environments, rewards functions were computed using Maximum entropy for a linear approximator and Deep Maximum Entropy for a complex, non-linear reward function.

## Requirements

* PyTorch

## Contents
  - [x] GridWorld Env
  - [x] ObjectWorld Env
  - [x] Maximum Entropy [1]
  - [x] Deep Maximum Entropy [2]

  
## Experiments

### GridWorld Env 
This environment is a rectangular grid. The cells of the grid correspond to the states of the environment. At each cell, four actions are possible: north, south, east, and west, which deterministically cause the agent to move one cell in the respective direction on the grid. Action would take the agent off the grid and leave its location unchanged. [[source]](https://tomaxent.com/2017/05/29/The-GridWorld-problem/#:~:text=A%20reinforcement%20learning%20task%20that,decision%20process%20(finite%20MDP).)
 
![real-rewards](https://user-images.githubusercontent.com/33734646/199789815-33d123db-b632-4bb7-ba8d-043494212db0.png)

#### MaxEnt

https://user-images.githubusercontent.com/33734646/199789457-48443e1a-66f5-479a-9722-daa4db997e09.mp4


#### Deep MaxEnt

https://user-images.githubusercontent.com/33734646/199789427-09096b89-ac14-4823-b294-b46f83025f7d.mp4



### ObjectWorld Env
The objectworld is an N×N grid of states with five actions per state, corresponding to steps in each direction and staying in place. Each action has a 30% chance of moving in a different random direction. Randomly placed objects populate the objectworld, and each is assigned one of C inner and outer colors. Object placement is randomized in the transfer environments, while N and C remain the same. There are 2C continuous features, each giving the Euclidean distance to the nearest object with a specific inner or outer color. In the discrete feature case, there are 2CN binary features, each one an indicator for a corresponding continuous feature being less than d ∈ {1, ...,N}.
The true reward is positive in states that are both within 3 cells of outer color 1 and 2 cells of outer color 2, negative within 3 cells of outer color 1, and zero otherwise. Inner colors and all other outer colors are distractors. [3]

#### MaxEnt
![ow-real-me](https://user-images.githubusercontent.com/33734646/200056281-3371b8d9-3668-4138-8d37-6240378ef150.png)

https://user-images.githubusercontent.com/33734646/200056327-7095c651-9990-48c1-ad3f-2a1fdb868371.mp4


#### Deep MaxEnt
![OW-real](https://user-images.githubusercontent.com/33734646/200056213-7f0d2aa1-6959-4e2b-be5c-1348cd334892.png)

https://user-images.githubusercontent.com/33734646/200056235-5620066a-5308-4305-9702-7b3e796c6019.mp4



## References
1. Thanh, H. V., An, L. T. H. & Chien, B. D. Maximum Entropy Inverse Reinforcement Learning. in Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) vol. 9622 661–670 (2016).
2. Wulfmeier, M., Ondruska, P. & Posner, I. Maximum Entropy Deep Inverse Reinforcement Learning. (2015). [arxiv](http://arxiv.org/abs/1507.04888)
3. Levine, S., Popović, Z. & Koltun, V. Nonlinear inverse reinforcement learning with Gaussian processes. Adv. Neural Inf. Process. Syst. 24 25th Annu. Conf. Neural Inf. Process. Syst. 2011, NIPS 2011 1–9 (2011).
