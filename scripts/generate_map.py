import json
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import CubicSpline

class Road_shapes_utils():
    def __init__(self,status,scale_1, scale_2, x_map_shift, y_map_shift, size_factor,angle_shift):
        self.status, self.scale_1, self.scale_2, self.x_map_shift, self.y_map_shift, self.size_factor,self.angle_shift=status, scale_1, scale_2, x_map_shift, y_map_shift, size_factor,angle_shift


    def generate_road_margins(self,road_points, offset):
        left_margins = []
        right_margins = []

        num_points = len(road_points)

        # Calculate the direction vectors for each road segment
        direction_vectors = []
        for i in range(num_points - 1):
            dx = road_points[i + 1][0] - road_points[i][0]
            dy = road_points[i + 1][1] - road_points[i][1]
            mag = np.sqrt(dx ** 2 + dy ** 2)
            direction_vectors.append((dx / mag, dy / mag))

        # Average neighboring direction vectors to get smoother normals
        averaged_directions = []
        for i in range(num_points - 1):
            if i == 0:
                averaged_directions.append(direction_vectors[0])
            elif i == num_points - 2:
                averaged_directions.append(direction_vectors[-1])
            else:
                averaged_directions.append(((direction_vectors[i][0] + direction_vectors[i - 1][0]) / 2,
                                            (direction_vectors[i][1] + direction_vectors[i - 1][1]) / 2))

        # Calculate normals and generate margins
        for i in range(num_points - 1):
            dx, dy = averaged_directions[i]
            nx = -dy
            ny = dx

            left_x = road_points[i][0] + offset * nx
            left_y = road_points[i][1] + offset * ny
            right_x = road_points[i][0] - offset * nx
            right_y = road_points[i][1] - offset * ny

            left_margins.append([left_x, left_y])
            right_margins.append([right_x, right_y])

        return left_margins, right_margins
    
    def create_non_closed_road(self,intermediate_points,start_angle,start_offset):
            for_conversions=For_convertion_utils(self.size_factor,self.x_map_shift,self.y_map_shift,self.angle_shift)
            starting_x_real, starting_y_real, starting_angle_real = intermediate_points[0][0],intermediate_points[0][1], start_angle
            if start_offset:
                initial_pose_real=[starting_x_real+start_offset[0], starting_y_real+start_offset[1], starting_angle_real]
            else:
                initial_pose_real=[starting_x_real, starting_y_real, starting_angle_real]
            initial_pose_sim=for_conversions.real2sim_xyp(initial_pose_real)
            scaled_vertices = []
            for point in intermediate_points:
                x,y,_= for_conversions.real2sim_xyp([point[0],point[1],0])
                scaled_vertices.append([x, y])

            # Interpolate a smooth road using cubic spline
            scaled_vertices = np.array(scaled_vertices)
            cs = CubicSpline(np.arange(len(scaled_vertices)), scaled_vertices)

            # Generate interpolated points along the smooth road
            num_interpolation_points = len(scaled_vertices) * 10  # Adjust as needed
            interpolated_points = cs(np.linspace(0, len(scaled_vertices) - 1, num_interpolation_points))

            # Format the road segments
            road_segments = []
            road_points =[]
            for point in interpolated_points:
                x, z = point
                road_points.append([x,z])
                y = 0.6  # Assuming constant y value
                road_segment = f"{x:.2f},{y:.2f},{z:.2f}"
                road_segments.append(road_segment)
            self.status.real_initial_waypoint=[initial_pose_real[0],initial_pose_real[1]]
            self.status.real_initial_angle=initial_pose_real[2]
            self.status.sim_initial_waypoint=[initial_pose_sim[0],initial_pose_sim[1]]
            self.status.sim_initial_angle=initial_pose_sim[2]
            return initial_pose_sim,initial_pose_real,road_points, '@'.join(road_segments)

    def create_closed_road(self, intermediate_points, start_angle, start_offset):
        for_conversions = For_convertion_utils(self.size_factor, self.x_map_shift, self.y_map_shift, self.angle_shift)
        starting_x_real, starting_y_real, starting_angle_real = intermediate_points[0][0], intermediate_points[0][1], start_angle
        
        # Determine the initial pose in the real world
        if start_offset:
            initial_pose_real = [starting_x_real + start_offset[0], starting_y_real + start_offset[1], starting_angle_real]
        else:
            initial_pose_real = [starting_x_real, starting_y_real, starting_angle_real]
        
        # Convert the initial pose to simulation coordinates
        initial_pose_sim = for_conversions.real2sim_xyp(initial_pose_real)
        
        # Convert intermediate points to simulation coordinates
        scaled_vertices = []
        for point in intermediate_points:
            x, y, _ = for_conversions.real2sim_xyp([point[0], point[1], 0])
            scaled_vertices.append([x, y])
        
        # Ensure smooth closure by wrapping the data and setting periodic boundary conditions
        scaled_vertices = np.array(scaled_vertices)
        n_points = len(scaled_vertices)
        
        # Duplicate the first point for smooth periodicity
        periodic_vertices = np.vstack([scaled_vertices, scaled_vertices[0]])
        
        # Create a cubic spline with periodic boundary conditions
        cs = CubicSpline(np.arange(len(periodic_vertices)), periodic_vertices, bc_type='periodic')
        
        # Generate interpolated points along the smooth road
        num_interpolation_points = n_points * 20  # Adjust for desired resolution
        interpolated_points = cs(np.linspace(0, n_points, num_interpolation_points, endpoint=False))
        
        # Format the road segments
        road_segments = []
        road_points = []
        for point in interpolated_points:
            x, z = point
            road_points.append([x, z])
            y = 0.6  # Assuming constant y value
            road_segment = f"{x:.2f},{y:.2f},{z:.2f}"
            road_segments.append(road_segment)
        
        # Update status
        self.status.real_initial_waypoint = [initial_pose_real[0], initial_pose_real[1]]
        self.status.real_initial_angle = initial_pose_real[2]
        self.status.sim_initial_waypoint = [initial_pose_sim[0], initial_pose_sim[1]]
        self.status.sim_initial_angle = initial_pose_sim[2]
        
        return initial_pose_sim, initial_pose_real, road_points, '@'.join(road_segments)

    def create_circular_road(self,radius_real):
        radius_sim=radius_real*self.size_factor

        center_x_sim=self.x_map_shift
        center_y_sim=self.y_map_shift

        road_definition,road_points = self.create_circle(center_x_sim,center_y_sim,radius_sim, 100)
        for_conversions=For_convertion_utils(self.size_factor,self.x_map_shift,self.y_map_shift,self.angle_shift)

        initial_pose_sim=self.x_map_shift+radius_sim,self.y_map_shift,0
        initial_pose_real=for_conversions.sim2real_xyp(initial_pose_sim)
        self.status.real_initial_waypoint=[initial_pose_real[0],initial_pose_real[1]]
        self.status.real_initial_angle=initial_pose_real[2]

        self.status.sim_initial_waypoint=[initial_pose_sim[0],initial_pose_sim[1]]
        self.status.sim_initial_angle=initial_pose_sim[2]
        return initial_pose_sim,initial_pose_real,road_points, road_definition


    def create_circle(self,start_x, start_z, radius, num_segments):
        road_segments = []
        road_segments_complete = []
        interpolated_points = []
        theta_values = np.linspace(0, 2*np.pi, num_segments)
        for theta in theta_values:
            x = start_x + radius * np.cos(theta)
            z = start_z + radius * np.sin(theta)
            interpolated_points.append([x,z])
            y = 0.6
            road_segment = f"{x:.2f},{y:.2f},{z:.2f}"
            road_segments.append(road_segment)
        # road_segments_complete.append(road_segments[-1])
        for road_segment in road_segments:
            road_segments_complete.append(road_segment)
        return '@'.join(road_segments_complete),interpolated_points
    
class For_convertion_utils():
    def __init__(self,size_factor,x_map_shift,y_map_shift,angle_shift):
        self.size_factor=size_factor
        self.x_map_shift=x_map_shift
        self.y_map_shift=y_map_shift
        self.angle_shift=angle_shift

    def real2sim_xyzypr(self,pose,orientation):
        x=pose[0]*self.size_factor+self.x_map_shift
        y=pose[1]*self.size_factor+self.y_map_shift
        angle=-orientation[2]+self.angle_shift
        return [x,y,angle]

    def sim2real_xyzypr(self,pose,orientation):
        x=(pose[0]-self.x_map_shift)/self.size_factor
        y=(pose[2]-self.x_map_shift)/self.size_factor
        angle=-(orientation[1]-self.angle_shift)
        return [x,y,angle]

    def real2sim_xyp(self,pose):
        x,y,p=pose
        x=x*self.size_factor+self.x_map_shift
        y=y*self.size_factor+self.y_map_shift
        angle=-p+self.angle_shift
        return [x,y,angle]

    def sim2real_xyp(self,pose):
        x,y,p=pose
        x=(x-self.x_map_shift)/self.size_factor
        y=(y-self.y_map_shift)/self.size_factor
        angle=-(p-self.angle_shift)
        return [x,y,angle]


class System_status_utils():
    def __init__(self):
        self.simulator_pose=[0.,0.,0.]
        self.simulator_orientation=[0.,0.,0.]

        self.tracked_pose_sim=[0.,0.,0.]

        self.real_pose=[0.,0.,0.]
        self.real_orientation=[0.,0.,0.]

        self.real_initial_waypoint=[0.,0.]
        self.real_initial_angle=0.

        self.sim_initial_waypoint=[0.,0.]
        self.sim_initial_angle=0.

        self.current_waypoint_index=0
        self.waypoint_size=0

        self.real_target_waypoint=None
        self.sim_target_waypoint=None

        self.real_waypoint_list=None
        self.sim_waypoint_list=None
        self.sim_waypoint_list_center=None
        self.sim_waypoint_list_right=None
        self.sim_waypoint_list_left=None


        self.real_image=None
        self.sim_image=None
        self.blended_image=None

        self.simulator_cte=0.0

        self.sim_image_for_model=None

        self.real_throttle_multiplier=0.3
        self.real_steering_multiplier=1

        self.state = "stopped"


        self.is_running=True

        self.keyboard_client=False
        self.real_camera_client=False
        self.tracking_client=False
        self.simulator_client=False
        self.real_car_controls=False
        self.feed_server=False
        self.simulator_client_object=None

        self.for_conversions=None

        self.collision=None

        self.waypoint_state="going_to_waypoint"
        self.waypoint_distance=0.0

        self.car_controls=[0.0,0.0,False]


        self.current_lane="left"


        self.status_obstacle_list=[]

        self.left_margins, self.right_margins = None,None

        self.real_speed=0.0
        self.simulated_speed=0.0

        self.speed_threshold_min=0.4

    

# EXAMPLES OF EXISTING ROADS

# output_filename = "map1.json"
# road_points=[[-1.616,-0.312],[-0.470,-1.500],[0.488,-1.111],[1.450,-0.303],[1.416,0.576],[0.558,0.763],[-0.291,0.687],[-0.764,1.192],[-1.562,1.133]]
# start_angle=-140
# start_offset_real=None
# closed_road=True

# output_filename = "map2.json"
# road_points=[[-1.616,-0.312],[-0.470,-1.500],[0.488,-1.111],[1.3,-0.8],[1.,0.5],[-0.291,0.687],[-0.764,1.192],[-1.562,1.133]]
# start_angle=-140
# start_offset_real=None
# closed_road=True

# output_filename = "map3.json"
# road_points=[[-1.616,-0.312],[-0.470,-1.500],[0.5,-1.7],[0.9,-1.3],[1.5,-0.5],[3,0.5],[3,1.2],[3,2],[2,2.5],[1,2],[0.2,1.5],[-0.4,1],[-1,1],[-1.5,0.5]]
# start_angle=-140
# start_offset_real=None
# closed_road=True

# output_filename = "map4.json"
# road_points=[[-8.,0.],[-6.,0.],[-4.,0.],[-2.,0.],[0.,0.],[1.5,0.],[2.2,0.4],[2.6,1.0],[2.7,1.5],[2.8,2.],[2.8,2.5],[2.7,3.],[2.6,3.5],[2.2,4.1],[1.5,4.5],[0,4],[-2.0,4.5],[-4.0,5],[-6.0,4.5],[-8.0,4.5],[-8.7,4.1],[-9.1,3.5],[-9.2,3.],[-9.3,2.5],[-9.3,2.],[-9.2,1.5],[-9.1,1.0],[-8.6,0.4]]
# start_angle=-140
# start_offset_real=None
# closed_road=True

# output_filename = "map5.json"
# road_points=[[-0.4,-0.4],[0.5,1.],[1.5,1.6],[2,2.5],[0.3,3.5],[1,5.],[3.5,4.5],[3.,2.],[2.,0.7],[1.5,0.],[1.5,-1.],[0.5,-2.],[-0.5,-1.5]]
# start_angle=0
# start_offset_real=None
# closed_road=True

# output_filename = "map6.json"
# road_points=[[-1.6,-0.7],[-1.5,-1.5],[-0.6,-1.5],[0.,-0.5],[1.1,-0.6],[1.4,0.5],[0.6,0.8],[-0.3,0.7],[-0.8,1.3],[-1.6,1.1],[-1.5,0.2]]
# start_angle=-180
# start_offset_real=None
# closed_road=True

# output_filename = "simple_straight.json"
# road_points=[[-1.5,0.],[-1.,0.],[-0.5,0.],[0.,0.],[0.5,0.],[1.,0.],[1.5,0.]]
# start_angle=-180
# start_offset_real=None
# closed_road=False

# END OF EXAMPLES OF EXISTING ROADS



STEERING = 0
THROTTLE = 1
SIZE_FACTOR=7.33
X_MAP_SHIFT=48
Y_MAP_SHIFT=50
ANGLE_SHIFT=90
SCALE_1=2.08
SCALE_2=3.2
WAYPOINTS_TRESHOLD = 0.4 * SIZE_FACTOR
ANGLE_TRESHOLD=5
OFFSET=3

def generate_road(road_generation,road_points,start_angle,start_offset_real,closed_road,output_filename):
    if closed_road:
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_closed_road(road_points,start_angle,start_offset_real)
    else:
        initial_pose_sim,initial_pose_real,sim_waypoint_list,road_definition=road_generation.create_non_closed_road(road_points,start_angle,start_offset_real)
    left_lane, right_lane = road_generation.generate_road_margins(sim_waypoint_list, 1.5)
    left_margins, right_margins = road_generation.generate_road_margins(sim_waypoint_list, OFFSET)
    if closed_road:
        left_lane.append(left_lane[0])
        right_lane.append(right_lane[0])
        left_margins.append(left_margins[0])
        right_margins.append(right_margins[0])

    sim_x, sim_y = zip(*[((x-X_MAP_SHIFT) / SIZE_FACTOR, (y-Y_MAP_SHIFT) / SIZE_FACTOR) for x, y in sim_waypoint_list])
    left_lane_x, left_lane_y = zip(*[((x-X_MAP_SHIFT) / SIZE_FACTOR, (y-Y_MAP_SHIFT) / SIZE_FACTOR) for x, y in left_lane])
    right_lane_x, right_lane_y = zip(*[((x-X_MAP_SHIFT) / SIZE_FACTOR, (y-Y_MAP_SHIFT) / SIZE_FACTOR) for x, y in right_lane])
    left_margin_x, left_margin_y = zip(*[((x-X_MAP_SHIFT) / SIZE_FACTOR, (y-Y_MAP_SHIFT) / SIZE_FACTOR) for x, y in left_margins])
    right_margin_x, right_margin_y = zip(*[((x-X_MAP_SHIFT) / SIZE_FACTOR, (y-Y_MAP_SHIFT) / SIZE_FACTOR) for x, y in right_margins])
    print("RG:  ",road_definition)
    data = {
        "initial_pose_sim": initial_pose_sim,
        "initial_pose_real": initial_pose_real,
        "sim_waypoint_list": sim_waypoint_list,
        "road_definition": road_definition,
        "left_lane": left_lane,
        "right_lane": right_lane,
        "left_margin": left_margins,
        "right_margin": right_margins,
        "obstacles": []
    }

    x_marks=[x for x,y in road_points]
    y_marks=[y for x,y in road_points]

    with open(output_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"JSON file '{output_filename}' has been created successfully.")

    plt.figure(figsize=(10, 8))
    plt.plot(sim_x, sim_y, 'b-', label="Simulated Waypoints")
    plt.plot(left_lane_x, left_lane_y, 'g-', label="Left Lane")
    plt.plot(right_lane_x, right_lane_y, 'r-', label="Right Lane")
    plt.plot(left_margin_x, left_margin_y, 'c--', label="Left Margin")
    plt.plot(right_margin_x, right_margin_y, 'm--', label="Right Margin")
    plt.plot((initial_pose_sim[0]-X_MAP_SHIFT) / SIZE_FACTOR, (initial_pose_sim[1]-Y_MAP_SHIFT)/ SIZE_FACTOR, 'ko', label="Pose", markersize=10)
    plt.plot(x_marks,y_marks, 'gx', label="waypoints", markersize=7)

    plt.title("Path and Lane Visualizations")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    min_val=min(min(min(left_margin_x),min(right_margin_x)),min(min(left_margin_y),min(right_margin_y)))
    max_val=max(max(max(left_margin_x),max(right_margin_x)),max(max(left_margin_y),max(right_margin_y)))
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    start_offset_real=None

    status = System_status_utils()
    for_conversions=For_convertion_utils(SIZE_FACTOR,X_MAP_SHIFT,Y_MAP_SHIFT,ANGLE_SHIFT)
    status.for_conversions=for_conversions
    road_generation=Road_shapes_utils(status,SCALE_1, SCALE_2, X_MAP_SHIFT, Y_MAP_SHIFT, SIZE_FACTOR, ANGLE_SHIFT)

    # PARAMETERS
    # Name of map file
    output_filename = "road_totest2.json"

    # List of waypoints
    
    #road_points=[ [0.0,1.2], [-1.5,1.2], [-2.1,0.0], [-1.5,-0.9], [0.0,-0.9], [1.5,-0.9], [2.1,0.0], [1.5,1.2] ]
    #road_points=[ [0.0,0.0], [0.0,1.0],[0.0,2.0],[0.0,3.0], [3.0,2.0]]  
    #road_points = [[0.0,0.0], [0.0,1.0], [-1.0, 2.0],[-2.0,1.0], [-2.0,0.0], [-2.0,-1.0],[-1.0,-2.0],[0.0,-1.0]]
    road_points = [[0.0,0.0], [0.0,5.0], [-5.0, 10.0],[-10.0,5.0], [-10.0,0.0], [-10.0,-5.0],[-5.0,-10.0],[0.0,-5.0]]

    # road_points=road_points[::-1]
    # Starting orientation
    start_angle=90

    # Is the road closed? (For example a circle)
    closed_road=True
    # END PARAMETER


    generate_road(road_generation,road_points,start_angle,start_offset_real,closed_road,output_filename)

