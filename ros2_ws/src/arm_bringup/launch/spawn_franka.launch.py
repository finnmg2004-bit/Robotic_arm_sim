import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Find the arm description and turn the xacro into a URDF string
    desc_share = get_package_share_directory("franka_description")
    xacro_file = os.path.join(get_package_share_directory("arm_bringup"), "urdf", "fr3_gazebo.urdf.xacro")
    robot_description = xacro.process_file(xacro_file).toxml()

    # 2. Tell Gazebo where meshes live (the folder that CONTAINS franka_description)
    mesh_path = SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.dirname(desc_share))

    # 3. Start Gazebo with an empty world; -r means start running immediately
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": "-r empty.sdf"}.items(),
    )

    # 4. Publish the URDF on the /robot_description topic
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": robot_description, "use_sim_time": True}])

    # 5. Read /robot_description and create the arm in Gazebo
    spawn = Node(package="ros_gz_sim", executable="create",
                 arguments=["-topic", "robot_description", "-name", "franka"])

    # 6. Pass Gazebo's clock into ROS so use_sim_time works
    clock = Node(package="ros_gz_bridge", executable="parameter_bridge",
                 arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"])

    jsb = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"])
    arm = Node(package="controller_manager", executable="spawner", arguments=["arm_controller"])

    return LaunchDescription([mesh_path, gazebo, rsp, spawn, clock, jsb, arm])
