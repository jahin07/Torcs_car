
Deliverable 2 by T16
Xiyao Ma, Yu Wen

In this chapter, we will demonstrate reinforcement learning and Deep Deterministic Policy Gradient we adopted in our project.
Training a self-training driver to drive a car is not like a supervised learning or an unsupervised learning. Reinforcement learning is a field between supervised learning and unsupervised learning, since we do not teach it how to drive the car correctly, but let it learn on its own, which means that it can try some actions and get some error to revise its action. The idea here is that through trial and error, an agent learns and in a specific situation it chooses actions that rewards it and avoids actions that punish it. 
Google DeepMind Group proposed a very famous algorithm, called Deep Q-Network(DQN). They explained the key idea and principle of how a computer learned to play video games just via observing the screen images and receiving rewards when the game score gets increased. The brilliant outcome indicates that the algorithm they proposed is sound to deal with reinforcement learning [1].
However, here comes another issue, which is that the actions produced by DQN are discrete while the action we would apply in Self-Driving are continuous. Obviously, there was a method to make it adapt to Self-Driving, that one can just discretize the entire action space. For instance, you can discretize the steering wheel from -90 to +90 degrees into 36 states, which each contains 5 degrees, and discretize the acceleration from 0km/h2 to 100km/h2 into 25 states, which each contains 4km/h2, which equals to 900 combinations. This method seems apply easily, but it would bring a very unfeasible problem; when we want AI to do something more specific and precise, we need to divide the action space into much more states, which means much more combinations and much more complexity. This is not the situation programmer working in self-driving would not like to see. Hence, here we decide to exploit another very powerful algorithm called Deep Deterministic Policy Gradient (DDPG)[2], which combines Deterministic Policy Gradient Algorithm[7] and Deep Q-Network.
A.	Reinforcement Learning 
1)	Basic Idea
An Agent interacts within an Environment at a sequence of time steps t = 0,1,2, …. At each time step, the agent obtains some states st(st∈S) and choose an action at(at∈A), where A is the action set in st. Then, it will get an immediate reward rt+1(rt∈R), and transits to another state st+1[8]. The basic structure of Reinforcement Learning is shown in Fig.1. In fact, reinforcement learning is a field between supervised learning and unsupervised learning, since we do not teach the agent how to act correctly, but let it learn on its own, which means that it can try some actions and get some error to revise its action [9]. This is the famous trial-and-error scheme.

Fig.1. Structure of Reinforcement Learning
2)	Markov Decision Process
The set of states and actions, together with rules and probabilities for transitioning from one state to another, make up a Markov Decision Process in Fig. 2[10].

Fig. 2. Markov Decision Process
One very important property of MDP is that the history of state transitions before a specific time step doesn’t matter. In other word, the future is not determined by the history but the current state, this property provides us a support to apply DQN and DDPG.
        (1)
B.	Deep Q-network
Dealing with the problem that play Atari 2600 video games, Google DeepMind proposed Deep Q-Network(DPN) in 2014, The performance in Atari is so remarkable that DQN is widely used in reinforcement learning. 
To represent the total reward after termination of the game, we define a function Q(s, a) with a discounted factor γ (0<=γ<=1), and fold the terms into γ times the Q value of the next time step in terms of Bellman Equation. 
(2)
The action-value function is used in many reinforcement learning algorithms. It describes the expected reward after taking an action at in state st and thereafter following policy π: 
(3)
If the target policy is deterministic, uses the greedy policy                                   of  to remove the expectation of target policy.
	 (4)
However, in policy-based methods, we directly try and estimate the action that could bring the max reward from the state, without first trying to explicitly find value of the state. A policy and the action it gives us might be easier to specify than first trying to assign a Q value. 
Then, we set up the objective that we need to optimize, the loss function with mean squared error (MSE) uses the principle of one-step temporal difference (TD) error instead of Monte-Carlo, so that calculate gradient descent in one time step and update the parameters in the actor-critic network quickly and efficiently.
(5)
where (6)
is a separate different target network, which though it has the same structure as Value function. We will talk about it later in detail.
1)	Actor-Critic Algorithm
Essentially, the Actor-Critic Algorithm is a hybrid method, which combines the policy gradient method and the value function method together. The policy function is like the actor, while the value function is referred to as the critic. For instance, the actor produces the action a based on the current state of the environment s, and the critic produces a TD error signal to judge the actions made by the actor [11], as shown in Fig. 3. 

Fig. 3 Actor-Critic Algorithm
			(7)
In addition, when evaluating the model, DeepMind found that the Actor-Critic Algorithm is not stable in some situations, so they proposed two methods to address it, which is to exploit a separate target Q-network in Loss Function presented above and using experience replay in training.
2)	 Separate Target Network
People could notice that network being updated is also used in calculating the target network so that the it is intractable to converge. Therefore, DQN introducing another network. While the separate target network in Equation (6) has the same structure as Q-network, it has older parameters and was updated slowly so that it could avoid oscillations of Q-network and break correlations between Q-network and target to enhance the stability of Q-network [12]. 
3)	Experience Replay
Another critical trick to stabilize the Q-network is experience replay. During the training, we first create a a finite-sized cache, replay buffer, and then randomly sample the transitions from the Environment and store the tuple (st, at, st+1, at+1) into the replay buffer. We would abandon the oldest samples when the replay buffer is full. At each time step t, we randomly sample a small batch from replay buffer and update the parameters in Actor-Critic Algorithm with it. This method is used to break the correlations between different training episodes and avoid stuck into local minimum [1].
C.	Deep Deterministic Gradient Policy
	Deep Deterministic Gradient Policy(DDPG), is proposed by Google in 2016 and evolved from DPG to take advantage of DQN and apply it into continuous space action. As I said in the beginning of Chapter 3, it is intractable and inefficiently to discrete the continuous action space into numerous states with DQN, so it is better for DPPG to use in continuous action space.
1)		Modify actor-critic network
	By taking advantage of DQN, DDPG modify the actor-critic network and separate target Q-network to improve the stability further. In DDPG, we copy another actor network and critic network  separately to calculate the target values. To stabilize the network, we introduce a factor , where  to constrain the update very slowly via the following two equations, while this scheme delay the propagation of value function estimate for that it need to calculate another actor-critic network in every time step, but it improves the stability tremendously.
(8)
	Another difficulty in continuous action space is to explore the environment, in DDPG, we add a Noise to the action as , noise used is Ornstein-Uhlenbeck process which describes the velocity of a massive Brownian particle under the influence of friction. This is to generate temporally correlated exploration of the action space and the pseudocode of DDPG algorithm is shown in Fig. 4. 

Fig. 4. Pseudocode of DDPG algorithm
2)	 Deterministic Policy Gradient
	Deterministic Policy Gradient is proposed by David Silver in2014 in [7], the motivation is that DPG is more efficient than stochastic policy gradients and is especially useful in continuous control problems where the policy won't produce probabilistic results. 
The derivative of a bump-shaped function like Gaussian or Log-Normal will have a spike going up and then suddenly down. Though Q value may be smooth, this leads to very high variance when multiplied, as shown in Fig.5.

Fig. 5. Stochastic Policy Gradient
To get a more deterministic policy to be sure of the action. In the limiting case, the up-down spike selects two Q values to a little right and little left. This leads us to the gradient of Q in the limit the Gaussian becomes a delta function as presented in Fig. 6.

Fig. 6 Deterministic Policy Gradient
	“Deep” of DDPG means that we apply deep neural network for calculating Q value function and policy function. In Atari 2600 games experiment by DeepMind, they specificly appled Convolutional Neural Network with the last four game picture frames as the input[1], but in our project, we directly use the 29 sensors value provided by TORCS simulator as input rather than CNN with last 4 video frame as input, since the role of CNN is to extract feature that is used to produce the policy. IN addition, different from applying a softmax function on the output of the last layer of CNN to get a vector of relative probability of choosing an action in discrete action space, in continuous action space, we interpret the outputs as mean and elements of the covariance matrix. These parameters can then be used to set up a Gaussian for us to sample and get the desired continuous value [14]. 
D.	Strategies of The Track Racing 
Based on the idea in the book Going Faster!: Mastering the Art of Race Driving [15], we get the idea to train the agent more efficiently.
Basically, we could separate the track in to the straightaway and the corners. The track could be thought as several straightaways linked by several corners. For example, we separate a track in TORCS into straightaways and corners, the result is showed in Fig. 7. below.

Fig.7. Separation of the track
As we could see, the track consists of 12 straightaways which are linked by 13 corners. In the two kinds of track modes, we need to apply different strategies. 
Firstly, on the straightaways, we need to apply full accelerate until we have to brake for the next corners to avoid the maximal speed for the next corner.
Secondly, in the corner, we need to use up the maximal speed limit of the corner. The maximal speed is limited by the maximal radius of the corner, and the G-force that the tires could provide. We use the corner #9 in the track above as an example, the strategy is showed in Fig. 8. below.

Fig. 8. Radius of the corner
If we use the same car and the same tires, so the G-force that the tires could provide is a constant. And the centripetal force is equal to the mass of the car multiply the squaring of the speed then divide by the radius. The maximal centripetal force is equal to the G-force that the tires could provide, the mass of the car is also a constant, if we want to get the maximal speed, we need to get the maximal radius of the corner. 
The two lanes in red and white color in the Fig. 8 above is two strategies when passing the corner. Based on the analysis above, we need to drive in the red lane and pass through the Apex point to get the maximal radius. We could get the maximal speed by applying this method.
Finally, we exit the corner and go into the straightaways again, we use the same strategy again: we full accelerate the car until we have to brake for the next corners.
Also, there is a relation between the steering angle and the radius. The relationship is showed in Fig. 9.

Fig. 9 Relationship between the variables
When the radius is much larger than the wheel base of the car (a constant for the same car). The radius equals to the wheel base divides by the sine of the steering angle. The figure of the radius on different steering angle is showed in Fig. 10.

Fig. 10 Figure of function (R/Sin(θ))
In the Fig  above we could see that when the steering angle is 0, the radius is ∞ and when the steering angle is π/2, the minimal radius is equal to the wheel base. When the car is on the straightaway, the radius is thus ∞. 
Generally, we want to get the maximal speed on the straightaways and the maximal speed in corners. So, we need to get the accelerate limit on the straightaways and maximal radius in corners[16]. In the TORCS simulation, it’s hard to separate the straightaways and corners, so we just get the maximal axis speed and maximal radius. However, the radius function could not be used in the programs, so we used the modified function: Wheel base divides by the addition of the sine of steering angle and constant number 1. The figure of the function is showed below in Fig. 11.

Fig. 11 Figure of function (R/(Sin(θ)+1))
This function is suitable to use in the program to be a part of the reward function. We need to get the maximal value of the function to get the maximal radius.
Besides, on the straightaways, we want to get the maximal axis speed and reduce the radial speed. The idea is showed in Fig.12 below. We use V(Axis) minus V(Radial) to achieve our design.

Fig. 12 resolution of velocity
Also, we don’t want the car to go far away from the Axis or the car would easily crush out the track. So, we applied a term: minus V multiply the distance between the car and the Axis.
Finally, we want to achieve the maximal radius in the corner. Also, in the straightaways, the radius is also the maximal value. So, we add V multiply radius into the reward function. And the radius equals to the wheel base divides by the addition of 1 and the sine of steering angle. In summary, the reward function is showed below:  
R = Vx-Vy-V*Distance+V*radius     (9)
I.	Performance Evaluation
A.	Experimental Setting
In this project, we evaluate our model on Gym TORCS, with a computer with Intel Core i7-6700 CPU @3.4Ghz *8, a Nvidia Geforce GTX 1080 and Linux 14.04. We also installed Anaconda to create a virtual environment, so that we could install all the libraries we need, such as Tensorflow, Keras, Gym, Torcs and their dependences.
a)	Keras
Keras is an API for the high-level neural network which based on TensorFlow or Theano in Python programming language. Keras is designed to support rapid experiments which could transfer your ideas into products quickly. The Keras has a lot of advantages. Firstly, using Keras is easy to design the prototype of your model quickly, because the Keras is a simple highly modular tool which could be easily extent. Secondly, the Keras could support RNN and CNN, and the combination of RNN and CNN. It’s very useful in handling real problems. Finally, the Keras supports seamless handover technique, it could easily switch in GPU and CPU.
Generally, Keras is a user-friendly, modular, scalable neural network API which supports Python from 2.7 to 3.5. It’s very suitable for us to use in the self-driving model design [17].
b)	TensorFlow
TensorFlow is a open-source software lab which could be used in many machine learning problems such as sensing and NLP. Though it is designed to use on single computer, it could be used on multiple CPUs and GPUs (with or without CUDA support). 
TensorFlow provided us with a set of Python API and some C++, Java and Go API. It could be used on 64-bit Linux or MacOS on both desktop system and server. It could also be used on mobile system such as Android and iOS. The computing of TensorFlow is based on the stated data flow map.
A lot of teams and programs rely on TensorFlow. Over 50 teams use TensorFlow in Google products such as Gmail, Google Photos and Google Searching Engine [18].
c)	Gym
Gym, established by OpenAI, is a world-wide general open source AI training and evaluating platform on games, websites, and other apps. Gym could create a human-like AI agent which get the information from the screen and hand held I/O devices like keyboard and mouse and train the human-like AI on any jobs that human could do on computers. It is very useful in developing, comparing reinforcement learning algorithm. It works on an VNC remote desktop so it doesn’t need the code or API of the quest program. We could use this toolkit to train the model in racing game.
d)	TORCS
TORCS (The Open Racing Car Simulator), created by Eric Espié and Christophe Guionneau, is an open-source 3D multi-platform racing simulator written in C and C++. It has over 50 different cars and 20 tracks and it also supports a lot of AI to run the simulator [19].  
	Most importantly TORCS provides 19 various type of sensor input for self-driving, as shown in Table. 1 (a) and (b) and some available effectors shown in Table. 2.


(a) Part 1

(b)Part 2
Table. 1. Description of input sensor in TORCS

Table. 2. Description of the available effectors
First, we need to install Anaconda to create a virtual environment and install all dependent packages and libraries in it, like below;

and activate torcs virtual environment,

Then we need to install tensorflow(0.12),keras(1.1) with pip.
We install gym torcs and its dependencies following the instruction on  https://github.com/ugo-nama-kun/gym_torcs. 


B.	Performance Metrics
Not hitting the edge of the road is not good enough for our AI-driver. We still need to evaluate the stability of AI-driver without swing, and how fast it can complete a circle. 
Analyzing the results, we can compare the performance of our model and others on one track and then we can tell that if our model is better. We will compare its lap time to the human driver’s and try to find out whether the model could find the best driving line of the track.
Finally, we will run the trained AI on different tracks to see if it could handle different environment and try to improve the overall performance of the AI on different tracks.
C.	Experimental Results
We trained and tested our self-driving agent on Gym TORCS. In TORCS, we apply a reward function that provide a positive reward of velocity projected on the direction of the track at each time step t, minus the speed projected in the vertical direction of the direction of driving, minus the speed times the distance between car and edge and plus the speed times the radius when car steer, by doing this, we want to max the velocity projected on the direction of the track, minimize the swing and have a route with a larger radius during the curve [16].
Meanwhile, we define a penalty of 1 for collisions and 200 for out of track. Episodes were terminated if progress was not made along the track after 500 frames. We put our code on Github: https://github.com/Shawn617/Torcs_car .
In practice, we set our hyper-parameters and parameters in our network followed the suggestion provided in [1], as shown in Table. 3.
Learning rate for actor
0.0001
Learning rate for critic
0.001
Discounted factor 
0.99
Target network hyper-parameters
0.001
Table. 3. Parameters setting in Network.
During the training, we choose the mode that does not show the training process but some information on the screen, such as lap time, best lap time, top and min speed, and damage, as shown in Fig. 13, which will enhance the training speed.

Fig.13. Training process
After training the model for 3 different tracks for 5 days with pre-trained model for only one track, we successfully realize self-driving in TORCS as shown in Fig. 15. (a) and (b), and even a better route with a higher speed based on some strategies of driving. In addition, by measuring the lap time of our model and other’s model, we have a better performance, as shown in Fig. 14. I put the demo on Youtube: https://youtu.be/z19Um3KinKo .

Fig. 14 Performance Comparasion

(a) Street 1

(b) Alpine 2
Fig. 15. Testing

