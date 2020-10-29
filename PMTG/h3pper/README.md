# H3pper PMTG 
This is an implementation of a [Policies Modulating Trajectories Generators](https://arxiv.org/abs/1910.02812) architecture for my h3pper robot in PyBullet. The original PMTG paper used a trajectory generator that output positions over time. I am using a wrench generator instead. [Dan Lynch](https://robotics.northwestern.edu/people/profiles/students/lynch-dan.html) has done a lot of amazing work to create an optimal controller for monopeds. His controller outputs the wrench we should apply at the foot over time. His model is for monopeds moving in a plane. He helped me adapt his wrench generator for a PMTG style architecture.    

# Architecture Overview 
Here is the initial architecture. <br />
![](../media/flowchart.jpg) 


# Results 

