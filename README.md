# CMSC818B_MiniProject_2

Paper to be covered: Multi-Agent Coverage in Urban Environments  

Link to the paper: https://arxiv.org/pdf/2008.07436.pdf  


Team - Members:      
Jingxi Chen, Email: ianchen@terpmail.umd.edu  
Md Ishat E Rabban , Email:  ier@umd.edu           
Vishnu Dutt Sharma, Email: vishnuds@umd.edu      	  
Kulbir Singh Ahluwalia, Email: kulbir@umd.edu       



## How to use:

### Requirements
This uploaded files requires Visual Studio. We have used VS 2019 for building it. 
However, you can install OpenGL and run with your chice of development environemtn as well.


### Steps

1. Voronoi Partitioning
	1. Run `python3 voronoi.py 0 0 800 600 6`, where the argument represtent the rectangle boundary and teyh number of robots
2. Lawnmower Algorithm
	1. Copy the `voronoi.txt` file into `OpenGL_Visual_Studio/OpenGL/` folder
	2. Open the `OpenGL.sln` file in visual Studio
	3. Update Line 30 in `main.cpp` to ```#define inputfilename "voronoi.txt"```
	4. Build and Run the project

















