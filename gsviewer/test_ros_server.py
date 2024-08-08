import rospy
import numpy as np

from geometry_msgs.msg import Pose, PoseArray
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest

if_update_object = False
def empty_service_callback(request:EmptyRequest):
    global if_update_object
    rospy.loginfo("Update object pose")
    if_update_object = True
    return EmptyResponse()

if __name__ == "__main__":
    rospy.init_node('test_ros_server', anonymous=True)

    object_poses_puber = rospy.Publisher('/object_poses', PoseArray, queue_size=5)
    rospy.Service('/update_object', Empty, empty_service_callback)

    name_list = ['cup_blue']
    cam_quat = np.array([
        0.5164546874594773,
        -0.3072582510860977,
        -0.40867222441671797,
        0.6869162510313304
    ])
    cam_trans = np.array([-3.7493473769904986, -2.1194623525833425, 1.0])

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        obj_pose_arr = PoseArray()
        obj_pose_arr.header.stamp = rospy.Time.now()
        obj_pose_arr.header.frame_id = ""

        obj_pose_arr.header.frame_id += "camera;"

        cam_trans[2] += 0.01
        cam_trans[2] = min(cam_trans[2], 3.0)
        camera_pose = Pose()
        camera_pose.position.x = cam_trans[0]
        camera_pose.position.y = cam_trans[1]
        camera_pose.position.z = cam_trans[2]
        camera_pose.orientation.x = cam_quat[0]
        camera_pose.orientation.y = cam_quat[1]
        camera_pose.orientation.z = cam_quat[2]
        camera_pose.orientation.w = cam_quat[3]

        obj_pose_arr.poses.append(camera_pose)

        for name in name_list:
            obj_pose_arr.header.frame_id += "{};".format(name)

            p = Pose()
            p.position.x = 0.0
            p.position.y = 0.0
            p.position.z = 1.0
            p.orientation.x = 0.0
            p.orientation.y = 0.0
            p.orientation.z = 0.0
            p.orientation.w = 1.0
            obj_pose_arr.poses.append(p)

        if len(obj_pose_arr.header.frame_id):
            # remove the last ";"
            obj_pose_arr.header.frame_id = obj_pose_arr.header.frame_id[:-1]

        object_poses_puber.publish(obj_pose_arr)
        rate.sleep()
