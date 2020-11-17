from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse


#bound (b1 (up-left), b2 (up-right), b3 (down-right), b4 (down-left))
def reflect(raw, bound):



    refl_up = raw.copy()
    refl_down= raw.copy()
    refl_left = raw.copy()
    refl_right = raw.copy()


    #up
    refl_up[:, 1] = 2 * bound[1, 1] - raw[:, 1]

    # down
    refl_down[:, 1] = 2 * bound[0, 1] - raw[:, 1]

    # left
    refl_left[:, 0] = 2 * bound[0, 0] - raw[:, 0]
    # right
    refl_right[:, 0] = 2 * bound[1, 0] - raw[:, 0]

    res= np.concatenate((points,refl_up,refl_down,refl_left,refl_right), axis=0)


    return res


def check_in_bound(p, bound):
    epislon= 0.00001
    if (p[0] >= bound[0,0] - epislon ) & (p[0] <= bound[1,0] + epislon ) & (p[1] >= bound[0,1] - epislon) & (p[1] <= bound[1,1] + epislon):
        return True

    return False


def sortVertices(array):

    center= np.sum(array, axis= 0) / len(array)

    angles= [(array[i] - center) for i in range(len(array))]
    angles= [np.arctan2(angles[i][1], angles[i][0]) * 180 /np.pi for i in range(len(angles))]
    angles= np.array((angles))
    indices = np.argsort(angles)


    return array[indices]



parser = argparse.ArgumentParser()

parser.add_argument('lower_left_x', type=float, help="Lower left point of the boundary, for x coordinate")
parser.add_argument('lower_left_y', type=float, help="Lower left point of the boundary, for y coordinate")
parser.add_argument('upper_right_x', type=float, help= "Upper right point of the boundary, for for x coordinate")
parser.add_argument('upper_right_y', type=float, help= "Upper right point of the boundary, for for y coordinate")
parser.add_argument('num_agents', type=int, help= "Number of agents")




args = parser.parse_args()


bound= np.array([[args.lower_left_x, args.lower_left_y],[args.upper_right_x, args.upper_right_y]])
num_a= args.num_agents



xs= np.random.uniform(bound[0][0], bound[1][0], size= num_a)
ys= np.random.uniform(bound[0][1], bound[1][1], size= num_a)

points= np.array(list(zip(xs,ys)))



# #Rectangular boundary for the map
# bound= np.array([[-1,-1], [3,3]])

#input points
# points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],  [2, 0], [2, 1], [2, 2]])
#points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],  [2, 0]])
# points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]])

num_p= points.shape[0]


count = 0
dist= 1000




while dist > 0.0001 and count < 200:
    refl_points= reflect(points, bound )


    vor = Voronoi(refl_points)

    region_index= vor.point_region[0:num_p]
    regions= [vor.regions[i] for i in region_index]
    region_points= [vor.vertices[i] for i in regions]
    centers = np.array([(np.sum(region_points[i], axis= 0) / len(regions[i])) for i in range(num_p)])

    dist= np.sum(np.linalg.norm(np.array(points - centers), axis= 1))

    points= centers

    count += 1


region_index= vor.point_region[0:num_p]
regions= [vor.regions[i] for i in region_index]
region_points= [vor.vertices[i] for i in regions]

region_points= [np.round(region_points[i],2) for i in range(len(region_points))]





#-> save to txt
file = open(r"voronoi.txt","w")
num_region= len(region_index)
file.writelines(str(num_region))
file.write("\n")
for i in range(num_p):
    region_i= region_points[i]
    region_i= sortVertices(region_i)
    file.writelines(str(len(region_i)))
    file.write("\n")
    for j in range(len(region_i)):
        file.writelines(str(region_i[j][0]) + "\t" + str(region_i[j][1]))
        file.write("\n")


file.close()


# -> Visualization
vor.vertices= np.round(vor.vertices, 2)

new_vertices= vor.vertices[[check_in_bound(p, bound) for p in vor.vertices]]



# plt.plot(new_vertices[:,0], new_vertices[:, 1], 'ko', ms=8)
#
# plt.plot(points[:,0], points[:, 1], 'go', ms=8)


plt.plot(new_vertices[:,0], new_vertices[:, 1], 'ko')

plt.plot(points[:,0], points[:, 1], 'go')


for vpair in vor.ridge_vertices:
    if vpair[0] >= 0 and vpair[1] >= 0:
        v0 = vor.vertices[vpair[0]]
        v1 = vor.vertices[vpair[1]]
        # Draw a line from v0 to v1.

        # plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)

        if check_in_bound(v0, bound) and check_in_bound(v1, bound):
            plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2)


plt.show()
