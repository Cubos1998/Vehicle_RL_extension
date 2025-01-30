#!/usr/bin/env python3

import rospy
import json
import time
import websockets
import tf
import math

from std_msgs.msg import String, Bool, Int64
from geometry_msgs.msg import PoseStamped
from mixed_reality.msg import Control, WaypointList, Waypoint, SimPose




#TODO: move this to a config file
control_uri = "ws://team10.local:8887/wsDrive"
FIXED_THROTTLE = True
REAL_CONTROLS = False
# WAYPOINT_THRESHOLD = 5
WAYPOINT_THRESHOLD = 0.3*7.33
ANGLE_THRESHOLD = 0

class Waypoint_control_utils():
        def __init__(self,treshold,angle_treshold):
            self.waypoint_treshold=treshold
            self.angle_treshold=angle_treshold


        def angle_difference(self,a1, a2):
            diff = a2 - a1
            while diff <= -180:
                diff += 360
            while diff > 180:
                diff -= 360

            return diff
    
        def angle_extraction(self, x1, y1, z1, x2, y2, z2):

            # Calculate the distances between points
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            # Calculate the angles on each axis
            angle_x_axis = math.atan2(dy, dz)
            angle_y_axis = math.atan2(dx, dz)
            angle_z_axis = math.atan2(dy, dx)

            # Convert angles from radians to degrees
            angle_x_axis_degrees = math.degrees(angle_x_axis)
            angle_y_axis_degrees = math.degrees(angle_y_axis)
            angle_z_axis_degrees = math.degrees(angle_z_axis)
            return angle_x_axis_degrees, angle_y_axis_degrees, angle_z_axis_degrees


        def calculate_control(self, x_target, y_target, simulator_pose, simulator_orientation):
            x_cur, _, y_cur = simulator_pose
            #print(f"Position\nx:{x_cur}, y:{y_cur}")
            _, angle_cur, _ = simulator_orientation
            #print(f"Orientation: {round(angle_cur, 3)}, {round(math.radians(angle_cur), 3)} rad")
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            #print(f"dist {distance}")
            _, angle_y_axis_degrees, _=self.angle_extraction(x_cur, 0.0, y_cur, x_target, 0.0, y_target)
            #print(f"Angle to goal: {angle_y_axis_degrees}")
            angle_difference=self.angle_difference(angle_cur,angle_y_axis_degrees)
            #print(f"angle diff {angle_difference}")
            steering = (math.radians(angle_difference) / math.pi)*5
            throttle=1

            return steering, throttle, distance, angle_difference
        
        def calculate_distance(self, x_target, y_target, simulator_pose):
            x_cur, _, y_cur = simulator_pose
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            return distance
        
        def calculate_angle(self, x_target, y_target, simulator_pose, simulator_orientation):
            x_cur, _, y_cur = simulator_pose
            _, angle_cur, _ = simulator_orientation
            _, angle_y_axis_degrees, _=self.angle_extraction(x_cur, 0.0, y_cur, x_target, 0.0, y_target)
            angle_difference=self.angle_difference(angle_cur,angle_y_axis_degrees)
            return angle_difference
        
        
        def rotate_to_point_start(self,point):
            print("[CAR WAYPOINT CONTROL] Rotating to point")
            i=0
            x,y=point
            angle_difference = self.calculate_angle(x,y)
            list_commands_thr=[1,-1]
            if angle_difference>0:
                    list_commands_str=[0.9,-0.9]
            else:
                    list_commands_str=[-0.9,0.9]
            while(abs(angle_difference)>self.angle_treshold and self.status.is_running and self.status.waypoint_state=="going_to_start"):
                print("Rotation angle remaining: ",int(angle_difference))
                time.sleep(.1)
                controls=(list_commands_thr[i],list_commands_str[i])
                self.status.car_controls=[controls[0],controls[1],False]
                time.sleep(0.5)
                if not self.status.real_car_controls:
                    self.status.car_controls=[controls[0],controls[1],True]
                    time.sleep(0.3)
                self.status.car_controls=[0,0,False]
                angle_difference = self.calculate_angle(x,y)
                time.sleep(0.3)
                if i==0:
                    i=1
                else:
                    i=0
            print("[CAR WAYPOINT CONTROL] Rotation complete")

        def rotate_to_angle_start(self,angle):
            print("[CAR WAYPOINT CONTROL] Rotating to angle")
            i=0
            _, angle_cur, _ = self.status.simulator_orientation
            angle_difference=self.angle_difference(angle_cur,angle)
            list_commands_thr=[1,-1]
            print(angle_difference)
            if angle_difference>0:
                    list_commands_str=[0.9,-0.9]
            else:
                    list_commands_str=[-0.9,0.9]
            while(abs(angle_difference)>self.angle_treshold and self.status.is_running):
                print("Rotation angle remaining: ",int(angle_difference))
                time.sleep(.1)
                controls=(list_commands_thr[i],list_commands_str[i])
                self.status.car_controls=[controls[0],controls[1],False]
                time.sleep(0.5)
                if not self.status.real_car_controls:
                    self.status.car_controls=[controls[0],controls[1],True]
                    time.sleep(0.3)
                self.status.car_controls=[0,0,False]
                _, angle_cur, _ = self.status.simulator_orientation
                angle_difference=self.angle_difference(angle_cur,angle)
                time.sleep(0.3)
                if i==0:
                    i=1
                else:
                    i=0
            print("[CAR WAYPOINT CONTROL] Rotation complete")
        
        def reach_start_loop(self, simulator_pose, simulator_orientation):    
            x,y=self.status.sim_initial_waypoint
            steering, throttle, distance, _ = self.calculate_control(x,y, simulator_pose, simulator_orientation)
            print("[CAR WAYPOINT CONTROL] Reaching start")
            while self.status.waypoint_state=="going_to_start" and self.status.is_running:
                if distance <= self.waypoint_treshold and self.status.waypoint_state!="reached_start":
                    self.status.car_controls=[throttle,steering,True]
                    self.status.set_waypoint_state("reached_start")
                    # self.status.set_state("stopping")
                    print("[CAR WAYPOINT CONTROL] REACHED START")
                elif self.status.waypoint_state!="reached_start":
                    self.status.car_controls=[throttle,steering,False]
                else:
                    self.status.car_controls=[0,steering,False]
                steering, throttle, distance, _ = self.calculate_control(x,y, simulator_pose, simulator_orientation)
                time.sleep(0.01)
            print("[CAR WAYPOINT CONTROL] Reached start")

        def go_to_waypoint_step(self, simulator_pose, simulator_orientation):    
            x,y=self.status.sim_target_waypoint
            steering, throttle, distance, _ = self.calculate_control(x,y, simulator_pose, simulator_orientation)
            if distance <= self.waypoint_treshold and self.status.waypoint_state!="reached_waypoint":
                self.status.car_controls=[throttle,steering,True]
                self.status.set_waypoint_state("reached_waypoint")
                self.status.set_state("stopping")
                real_waypoint=self.status.for_conversions.sim2real_xyp([x,y,0])
                print(f"[CAR WAYPOINT CONTROL] Reached waypont (SIM): {x}, {y} (REAL): {real_waypoint[0]}, {real_waypoint[1]}")
            elif self.status.waypoint_state!="reached_waypoint":
                self.status.car_controls=[throttle,steering,False]
                time.sleep(0.1)
            else:
                self.status.car_controls=[0,0,False]
                time.sleep(0.1)


        def go_to_start_logic(self, simulator_pose, simulator_orientation):
            print("[CAR WAYPOINT CONTROL]: Recieved a starting sequence request")
            initial_x,initial_y=self.status.sim_initial_waypoint[0],self.status.sim_initial_waypoint[1]
            _, angle_cur, _ = self.status.simulator_orientation
            if self.calculate_distance(initial_x,initial_y)<=self.waypoint_treshold:
                if self.angle_difference(angle_cur,self.status.sim_initial_angle)<=self.angle_treshold:
                    print("[CAR WAYPOINT CONTROL]: Already in start pose")
                    self.status.set_waypoint_state("reached_start")
                    self.status.set_state("driving")
                else:
                    print("[CAR WAYPOINT CONTROL]: Already in start pose but wrong angle")
                    self.rotate_to_angle_start(self.status.sim_initial_angle)
                    self.status.set_waypoint_state("reached_start")
                    time.sleep(1)
                    self.status.set_state("driving")
            else:
                self.rotate_to_point_start(self.status.sim_initial_waypoint)
                self.reach_start_loop(simulator_pose, simulator_orientation)
                time.sleep(2)
                self.status.car_controls=[0.,0.,True]
                self.rotate_to_angle_start(self.status.sim_initial_angle)
                self.status.set_state("stopping")


class Car_control_utils():
    def __init__(self,uri,fixed_throttle,real_controls):
        self.websocket=None
        self.uri=uri
        self.real_controls=real_controls
        #self.sim_client=status.simulator_client_object
        self.fixed_throttle=fixed_throttle



    async def send_command_to_car(self, throttle, steering):
        command = {
            "throttle": throttle,
            "angle": steering
        }
        message = json.dumps(command)
        await self.websocket.send(message)

    def send_command_to_sim(self,throttle,steering,brake):
        if brake:
            message = { 'msg_type' : 'control', 'steering': steering.__str__(), 'throttle':'0.0', 'brake': '1.0' }
        else:
            message = { 'msg_type' : 'control', 'steering': steering.__str__(), 'throttle':throttle.__str__(), 'brake': '0.0' }    
        
        self.sim_client.queue_message(message)

    async def connect_and_control_real_car(self):
        if self.real_controls:
            async with websockets.connect(self.uri) as websocket:
                self.websocket=websocket
                await self.car_control_loop()
        else:
            await self.car_control_loop()

    async def connect_and_control_obstacle(self):
        
        await self.obstacle_control_loop()

    async def car_send_commands(self,throttle,steering):
        throttle=throttle*self.status.real_throttle_multiplier
        steering=steering*self.status.real_steering_multiplier
        
        if steering>1:
            steering=0.99
        if steering<-1:
            steering=-0.99
        if self.status.state=='stopping':
            self.status.set_state('stopped')
            if self.real_controls:
                await self.send_command_to_car(-throttle, steering)
                # self.send_command_to_sim(0, steering, False)
                self.send_command_to_sim(-throttle, steering,True)
                time.sleep(0.3)
                await self.send_command_to_car(0., steering)
            else:
                self.send_command_to_sim(-throttle, steering,True)
        elif self.status.state=='stopped':
            if self.real_controls:
                await self.send_command_to_car(0., steering)
                # self.send_command_to_sim(0., 0.,False)
                self.send_command_to_sim(0., 0.,True)
            else:
                self.send_command_to_sim(0., 0.,True)
        else:
            if self.real_controls:
                real_throttle=throttle
                if throttle<0:
                    real_throttle=throttle*3
                await self.send_command_to_car(real_throttle, steering)
                self.send_command_to_sim(throttle, steering, False)
                # self.send_command_to_sim(0.0, steering, False)
            else:
                self.send_command_to_sim(throttle, steering, False)

    async def car_send_brake_command(self,throttle,steering):
        if self.status.state!='stopping' and self.status.state!='stopped':
            if self.real_controls:
                await self.send_command_to_car(-throttle, steering)
                self.send_command_to_sim(0, steering,False)
                time.sleep(0.3)
                await self.send_command_to_car(0., steering)
            else:
                self.send_command_to_sim(-throttle, steering,True)

    
    async def car_control_loop(self):
        while self.status.is_running:
            try:
                throttle=self.status.car_controls[0]
                steering=self.status.car_controls[1]
                brake=self.status.car_controls[2]
                if self.status.collision!='none' and self.status.state!="stopped":
                    print("COLLISION!",self.status.collision)
                    self.status.set_state("stopping")
                if brake:
                    await self.car_send_brake_command(throttle,steering)
                else:
                    await self.car_send_commands(throttle,steering)
                time.sleep(0.05)
            except Exception as e:
                print(f"Error: {e}")

    async def obstacle_control_loop(self):
        while self.status.is_running:
            try:
                throttle=0.
                steering=0.
                brake=0.
                if self.status.collision!='none':
                    print("COLLISION!",self.status.collision)
                await self.car_send_commands(throttle,steering)
                time.sleep(0.02)
            except Exception as e:
                print(f"Error: {e}")



throttle = 1.0
steering = 0.0
brake = False
collision = "none"
going = False
state = "stopped"   #state can only be driving, stopping, or stopped
sim_pose = [0.,0.,0.]
sim_orientation = [0.,0.,0.]
waypoint_list = []
current_waypoint_index = 0
following_waypoints = False
pub_throttle_steering = None










def new_throttle_steering(msg):
    #having a list of waypoints to follow will be prioritised over receiving direct throttle and steering values from the model
    if not following_waypoints:
        global throttle
        global steering
        global brake
        throttle = msg.throttle
        steering = msg.steering
        brake = msg.brake

def new_collision(msg):
    global collision
    collision = msg.data

def new_going(msg):
    global going
    global state
    global brake
    global pub_throttle_steering
    if going and msg.data:
        print(f"CONTROL: State is already driving")
    elif not going and not msg.data:
        print(f"CONTROL: State is already {state}]")
        if pub_throttle_steering is None:
            pub_throttle_steering = rospy.Publisher("control/throttle_steering", Control, queue_size=10)
        pub_throttle_steering.publish(Control(0, steering, True, False, True))
    elif msg.data:
        state = "driving"
        going = True
        brake = False
    else:
        state = "stopping"
        going = False
        if pub_throttle_steering is None:
            pub_throttle_steering = rospy.Publisher("control/throttle_steering", Control, queue_size=10)
        pub_throttle_steering.publish(Control(0, steering, True, False, True))
    
def new_waypoints(msg):
    global waypoint_list
    global current_waypoint_index
    global following_waypoints
    global state
    global going
    waypoint_list = list(map(lambda wp:[wp.x, wp.y],msg.waypoints))
    if msg.reset: current_waypoint_index = 0
    if len(waypoint_list)==0:
        following_waypoints = False
        state = "stopping"
    else:
        following_waypoints = True
        state = "driving"
        going = True
    print(f"Received waypoint list of length {len(waypoint_list)}\nCurrent waypoint index is {current_waypoint_index}")
    #print(waypoint_list)

def new_sim_pose(msg):
    global sim_pose
    global sim_orientation
    sim_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    quaternion = (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion) #this gives roll, pitch, yaw
    angle_rad = euler[1]
    if angle_rad<0:
        angle_rad = angle_rad + 2*math.pi
    sim_orientation = [math.degrees(euler[2]), math.degrees(angle_rad), math.degrees(euler[0])]    #sim_orientation is yaw, pitch, roll
    print(f"Angle: {round(sim_orientation[1], 3)}, {round(angle_rad, 3)} rad")
    print(f"Quaternion {quaternion}")

def new_euler(msg):
    global sim_pose
    global sim_orientation
    sim_pose = [msg.x,msg.y,msg.z]
    sim_orientation = [0,msg.yaw,0]
    
    




def control_node():
    print("Starting control node")
    global control_uri      #param
    global FIXED_THROTTLE   #param
    global REAL_CONTROLS    #param (maybe unnecessary)
    global following_waypoints #set by keyboard
    global throttle     #
    global steering
    global brake
    global collision
    global state
    global going
    global current_waypoint_index
    global waypoint_list
    global pub_throttle_steering
    global sim_pose
    global sim_orientation

    #TODO: get rid of TESTING
    TESTING = False
    
    rospy.init_node("control_node", anonymous=True)
    rospy.Subscriber("model/throttle_steering", Control, new_throttle_steering)
    rospy.Subscriber("collision", String, new_collision)
    rospy.Subscriber("going", Bool, new_going)
    rospy.Subscriber("waypoints", WaypointList, new_waypoints)
    #rospy.Subscriber("sim/pose", PoseStamped, new_sim_pose)
    rospy.Subscriber("sim/euler", SimPose, new_euler)
    if pub_throttle_steering is None:
        pub_throttle_steering = rospy.Publisher("control/throttle_steering", Control, queue_size=10)
    pub_current_waypoint_index = rospy.Publisher("control/current_waypoint_index", Int64, queue_size=10)
    #TODO: check if this rate is fine
    rate = rospy.Rate(4.4)

    waypoint_controller = Waypoint_control_utils(WAYPOINT_THRESHOLD, ANGLE_THRESHOLD)
    
    while not rospy.is_shutdown():
        try:
            if collision!="none" and state!="stopped":
                print("COLLISION!", collision)
                state = "stopping"
                going = False
                break

            if following_waypoints and going:
                #recalculate commands to next waypoint
                current_waypoint = waypoint_list[current_waypoint_index]
                print("Going to: ", current_waypoint)
                x, y = current_waypoint

                while abs(waypoint_controller.calculate_angle(x,y,sim_pose,sim_orientation))>90:
                    current_waypoint_index+=1
                    if current_waypoint_index >=len(waypoint_list):
                        print("Remaining waypoints are all behind the car")
                        state = "stopping"
                        following_waypoints = False
                        break
                    current_waypoint = waypoint_list[current_waypoint_index]
                    x, y = current_waypoint

                print("Going to: ", current_waypoint)
                x, y = current_waypoint

                distance = waypoint_controller.calculate_distance(x,y,sim_pose)
                if distance <= WAYPOINT_THRESHOLD:
                    current_waypoint_index += 1  # Move to the next waypoint
                    if  current_waypoint_index < len(waypoint_list):
                        current_waypoint = waypoint_list[current_waypoint_index]
                        print("Going to: ", current_waypoint)
                    else:
                        if TESTING:
                            print("reached final waypoint, looping")
                            current_waypoint_index = 0
                        else:
                            if going:
                                print("Reached final waypoint")
                            state = "stopping"
                            following_waypoints = False

                x, y = current_waypoint
                steering, throttle, _, _ = waypoint_controller.calculate_control(x, y, sim_pose, sim_orientation)
                pub_current_waypoint_index.publish(current_waypoint_index)

            if brake:
                #send brake command
                if state=="driving":
                    pub_throttle_steering.publish(Control(-throttle, steering, True, False, not going))
            else:
                #send commands
                if state=="stopping":
                    state = "stopped"
                    going = False
                    pub_throttle_steering.publish(Control(-throttle, steering, True, False, True))
                elif state=="stopped":
                    going = False
                    pub_throttle_steering.publish(Control(0, steering, True, False, True))
                else:
                    pub_throttle_steering.publish(Control(throttle, steering, False, throttle<0, not going))

            rate.sleep()
        except Exception as e:
            print(f"Error: {e}")
    


if __name__ == '__main__':
    try:
        control_node()
    except rospy.ROSInterruptException:
        pass