#!/home/leo/anaconda3/envs/py39/bin/python3

import rospy
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import TwistStamped, Twist
from sensor_msgs.msg import Imu
import time
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry
from mavros_msgs.msg import State
from std_msgs.msg import Float64MultiArray

class Px4Controller:

    def __init__(self, uavtype):

        self.type = uavtype
        self.imu = None
        self.current_state = State()
        self.cmd_vel = TwistStamped()
        self.local_cmd = TwistStamped()
        self.theta = Float64MultiArray()
        self.current_heading = None
        self.x = None
        self.y = None
        self.z = None
        self.desired_x = 0.0
        self.desired_y = 0.0
        self.desired_z = 0.5
        self.cur_target_pose = None
        self.arm_state = False
        self.offboard_state = False
        self.received_imu = False
        self.state = None

        '''
        ros subscribers
        '''
        self.state_sub = rospy.Subscriber(uavtype +"/mavros/state", State, self.state_cb, queue_size=10)
        self.imu_sub = rospy.Subscriber(uavtype +"/mavros/imu/data", Imu, self.imu_callback)
        self.odom_sub = rospy.Subscriber(uavtype +"/mavros/local_position/odom", Odometry, self.odom_cb, queue_size=10)
        '''
        ros publishers
        '''
        self.theta_pub = rospy.Publisher("/theta_"+uavtype, Float64MultiArray, queue_size=10)
        self.vel_pub = rospy.Publisher(uavtype +'/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        '''
        ros services
        '''
        self.armService = rospy.ServiceProxy(uavtype +'/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy(uavtype +'/mavros/set_mode', SetMode)
        print(uavtype + " Px4 Controller Initialized!")

    def state_cb(self, msg):
        self.current_state = msg

    def odom_cb(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z
        if rospy.get_param(self.type +"/vel_control") == 0:
            self.constHeight()
    
    def constHeight(self):
        self.local_cmd.twist.linear.x = (self.desired_x - self.x)
        self.local_cmd.twist.linear.y = (self.desired_y - self.y)
        self.local_cmd.twist.linear.z = (self.desired_z - self.z)
        self.vel_pub.publish(self.local_cmd)

    def OffboardandArm(self):
        self.arm_state = self.arm()
        self.offboard_state = self.offboard()
        
    def vel_control(self,msg):
        rospy.set_param(self.type +"/vel_control",1)
        self.cmd_vel.header.stamp = rospy.Time.now()
        self.cmd_vel.twist.linear.x = msg.linear.x
        self.cmd_vel.twist.linear.y = msg.linear.y
        self.cmd_vel.twist.linear.z = msg.linear.z
        self.cmd_vel.twist.angular.z = msg.angular.z
        self.vel_pub.publish(self.cmd_vel)

    def imu_callback(self, msg):
        self.imu = msg
        self.current_heading = self.q2yaw(self.imu.orientation)
        self.received_imu = True
        self.theta.data = [self.current_heading]
        self.theta_pub.publish(self.theta)
        
    def q2yaw(self, q):
        if isinstance(q, Quaternion):
            rotate_z_rad = q.yaw_pitch_roll[0]
        else:
            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad  
        
    def arm(self):
        if self.armService(True):
            return True
        else:
            print(self.type +"Vehicle arming failed!")
            return False

    def offboard(self):
        if self.flightModeService(custom_mode='OFFBOARD'):
            return True
        else:
            print(self.type +"Vechile Offboard failed")
            return False
            
if __name__ == '__main__':
    try:
        rospy.init_node('ArmandOffboard')
        uavtype = ["uav0","uav1","uav2","uav3","uav4","uav5"]
        px6_1 = Px4Controller(uavtype[0])
        px6_2 = Px4Controller(uavtype[1])
        px6_3 = Px4Controller(uavtype[2])
        px6_4 = Px4Controller(uavtype[3])
        px6_5 = Px4Controller(uavtype[4])
        px6_6 = Px4Controller(uavtype[5])
        px6_1.OffboardandArm()
        px6_2.OffboardandArm()
        px6_3.OffboardandArm()
        px6_4.OffboardandArm()
        px6_5.OffboardandArm()
        px6_6.OffboardandArm()
        last_time_1 = rospy.Time.now()
        last_time_2 = rospy.Time.now()
        last_time_3 = rospy.Time.now()
        last_time_4 = rospy.Time.now()
        last_time_5 = rospy.Time.now()
        last_time_6 = rospy.Time.now()
        while not rospy.is_shutdown():
            if rospy.Time.now() - last_time_1 > rospy.Duration(10):
                if not px6_1.current_state.armed or px6_1.current_state.mode != "OFFBOARD":
                	px6_1.OffboardandArm()
                	last_time_1 = rospy.Time.now()

            if rospy.Time.now() - last_time_2 > rospy.Duration(10):
                if not px6_2.current_state.armed or px6_2.current_state.mode != "OFFBOARD":
                	px6_2.OffboardandArm()
                	last_time_2 = rospy.Time.now()

            if rospy.Time.now() - last_time_3 > rospy.Duration(10):
                if not px6_3.current_state.armed or px6_3.current_state.mode != "OFFBOARD":
                	px6_3.OffboardandArm()
                	last_time_3 = rospy.Time.now()
            if rospy.Time.now() - last_time_4 > rospy.Duration(10):
                if not px6_4.current_state.armed or px6_4.current_state.mode != "OFFBOARD":
                    px6_4.OffboardandArm()
                    last_time_4 = rospy.Time.now()
            if rospy.Time.now() - last_time_5 > rospy.Duration(10):
                if not px6_5.current_state.armed or px6_5.current_state.mode != "OFFBOARD":
                    px6_5.OffboardandArm()
                    last_time_5 = rospy.Time.now()
            if rospy.Time.now() - last_time_6 > rospy.Duration(10):
                if not px6_6.current_state.armed or px6_6.current_state.mode != "OFFBOARD":
                    px6_6.OffboardandArm()
                    last_time_6 = rospy.Time.now()
    except rospy.ROSInterruptException:
        pass 

