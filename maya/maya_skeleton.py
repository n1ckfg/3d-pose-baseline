import json 
import maya.cmds as cmds
import pymel.core as pm
import maya.OpenMaya as om
import math
import re
import os


#path to imageplane content
openpose_images = "/home/flyn/git/3d-pose-baseline/test_images/" # replace it with abs path like "/path/to/bg_images" 
#path to 3d-pose-baseline
threed_pose_baseline = "/home/flyn/git/3d-pose-baseline/"
#for 3d use 3d_data.json and set three_dim to True
input_json_path = [os.path.join(threed_pose_baseline, "maya/{0}.json".format(_data)) for _data in ["3d_data", "2d_data"]] # replace it with abs path like "/path/to/2d_data.json" 


        
def load_data(data, threed):
    suffix = "threed" if threed else "twod"
    # jnts to ignore
    to_pass = [5,4,9,10,12] if threed else []
    # locator driver grp
    if not cmds.objExists("drivers_{0}".format(suffix)):
        cmds.group(n="drivers_{0}".format(suffix), em=True)
    for frame, jnt in data.iteritems():
        if not cmds.objExists("anim_joint"):
            cmds.group(n="anim_joint", em=True)
            anim_grp_prj = cmds.group(n="anim_joint_2d", em=True)
            cmds.parent(anim_grp_prj, "anim_joint")
        for jnt_id, trans in jnt.iteritems():
            if not int(jnt_id) in to_pass:
                if not cmds.objExists("anim_jnt_driver_{0}_{1}".format(jnt_id, suffix)):
                    cmds.select(clear=True)
                    jnt = cmds.joint(n="jnt_{0}_{1}".format(jnt_id, suffix), relative=True)
                    cmds.setAttr("{0}.radius".format(jnt), 10)
                    cmds.setAttr("{0}.displayLocalAxis".format(jnt), 1)
                    # match same pos for first frame
                    if threed:
                        cmds.move(trans["translate"][0],trans["translate"][1], trans["translate"][2], jnt)
                    else:
                        cmds.move(trans["translate"][0],trans["translate"][1], jnt)
                    anim_grp_child = cmds.listRelatives("anim_joint", children=True) or []
                    if not jnt in anim_grp_child:
                        cmds.parent(jnt, "anim_joint")

                    if threed:
                        #create 2d projection
                        jnt_proj = cmds.duplicate(jnt, n="jnt_prj_{0}".format(jnt_id))
                        cmds.pointConstraint(jnt, jnt_proj, mo=False, skip="z")
                        cmds.setAttr("{0}.translateZ".format(jnt_proj[0]), 0)
                        cmds.parent(jnt_proj, "anim_joint_2d")

                    # driver locator
                    driver = cmds.spaceLocator(n="anim_jnt_driver_{0}_{1}".format(jnt_id, suffix))
                    # drive jnt with animated locator frim frame 0
                    cmds.pointConstraint(driver, jnt)
                    #if not driver in cmds.listRelatives("drivers_{0}".format(suffix), children=True) or []:
                    cmds.parent(driver, "drivers_{0}".format(suffix))
                # add trans anim values to driver locator
                cmds.setKeyframe("anim_jnt_driver_{0}_{1}".format(jnt_id, suffix), t=frame, v=trans["translate"][0], at='translateX')
                cmds.setKeyframe("anim_jnt_driver_{0}_{1}".format(jnt_id, suffix), t=frame, v=trans["translate"][1], at='translateY')
                if threed:
                    cmds.setKeyframe("anim_jnt_driver_{0}_{1}".format(jnt_id, suffix), t=frame, v=trans["translate"][2], at='translateZ')
    # hacking 3d-pose-baseline coord. to maya
    cmds.setAttr("drivers_{0}.rotateX".format(suffix), -110 if threed else -180)

def parent_skeleton(jnt_mapping):
    if not isinstance(jnt_mapping, dict):
        raise Exception("expected dict, {0}".format(type(jnt_mapping)))
    #parent jnts based on jnt_mapping
    for body_part, jnt_map in jnt_mapping.iteritems():
        for map_dict in jnt_map:
            for parent_jnt, child_jnt in map_dict.iteritems():
                if isinstance(child_jnt, list):
                    for child in child_jnt:
                        cmds.parent(child, parent_jnt)
                else:
                    cmds.parent(child_jnt,parent_jnt)

def get_rotate(p1, p2):
    #calc rot for 3d json
    punkt_a = om.MPoint(p1[0], p1[1], p1[2])
    punkt_b = om.MPoint(p2[0], p2[1], p2[2])
    rot_vector = punkt_a - punkt_b
    world = om.MVector(0, 1, 0)
    quat = om.MQuaternion(world, rot_vector, 1) 
    mat = om.MTransformationMatrix()
    util = om.MScriptUtil()
    util.createFromDouble(0, 0, 0)
    rot_i = util.asDoublePtr()
    mat.setRotation(rot_i, om.MTransformationMatrix.kXYZ)
    mat = mat.asMatrix() * quat.asMatrix()
    quat = om.MTransformationMatrix(mat).rotation()
    m_rotation = om.MVector(math.degrees(quat.asEulerRotation().x),
                                        math.degrees(quat.asEulerRotation().y),
                                        math.degrees(quat.asEulerRotation().z)
                                        )
                                        
    return (m_rotation[0],m_rotation[1],m_rotation[2])
                      
def set_orient(data, jnt_mapping):
    #set orient 
    for frame, jnt in data.iteritems():
        cmds.currentTime(int(frame))
        for body_part, jnt_map in jnt_mapping.iteritems():
            for map_dict in jnt_map:
                for parent_jnt, child_jnt in map_dict.iteritems():
                    if not isinstance(child_jnt, list):
                        p1 = cmds.xform(parent_jnt, q=True, t=True, ws=True)
                        p2 = cmds.xform(child_jnt, q=True, t=True, ws=True)
                        rotation = get_rotate(p1,p2)
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[0], at='rotateX')
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[1], at='rotateY')
                        cmds.setKeyframe(parent_jnt, t=frame, v=rotation[2], at='rotateZ')


def main():
    threed_jnt_mapping = { 'root': [{'jnt_11_threed': ['jnt_1_threed','jnt_6_threed']},
                             {'jnt_13_threed': ['jnt_17_threed', 'jnt_25_threed']}],
                    #left leg from jnt 6 to 8
                    "left_leg":[{"jnt_{0}_threed".format(n):"jnt_{0}_threed".format(n+1)} for n in range(6,8)],
                    #right leg from jnt 1 to 3
                    "right_leg":[{"jnt_{0}_threed".format(n):"jnt_{0}_threed".format(n+1)} for n in range(1,3)],
                    #left arm from jnt 17 to 19
                    "left_arm":[{"jnt_{0}_threed".format(n):"jnt_{0}_threed".format(n+1)} for n in range(17,19)],
                    #right arm from jnt 25 to 27
                    "right_arm":[{"jnt_{0}_threed".format(n):"jnt_{0}_threed".format(n+1)} for n in range(25,27)]}

    twod_jnt_mapping = { 'root': [{'jnt_1_twod': ['jnt_11_twod','jnt_8_twod','jnt_5_twod', 'jnt_2_twod']}],
                    #left leg from jnt 11 to 13
                    "left_leg":[{"jnt_{0}_twod".format(n):"jnt_{0}_twod".format(n+1)} for n in range(11,13)],
                    #right leg from jnt 8 to 10
                    "right_leg":[{"jnt_{0}_twod".format(n):"jnt_{0}_twod".format(n+1)} for n in range(8,10)],
                    #left arm from jnt 5 to 7
                    "left_arm":[{"jnt_{0}_twod".format(n):"jnt_{0}_twod".format(n+1)} for n in range(5,7)],
                    #right arm from jnt 2 to 4
                    "right_arm":[{"jnt_{0}_twod".format(n):"jnt_{0}_twod".format(n+1)} for n in range(2,4)]}

    #read 2 or 3d json payload
    for _data in input_json_path:
        with open(_data) as json_data:
            # loaded data format:
            #   frames x jnts x (x,y,z)
            #   {frame<n>:[jnt<n>:"translate":[x,y,z], jnt<n>:"translate":[x,y,z], ...], frame<n>:...}
            data = json.load(json_data)
        #set orient on 3d
        if "3d_data" in _data:
            #load data and build locs and jnts
            load_data(data, True) #true for 3d data
            #parent jnts
            parent_skeleton(threed_jnt_mapping)
            #set orientation fo jnts
            set_orient(data, threed_jnt_mapping)
        else:
            #load data and build locs and jnts
            load_data(data, False)
            #parent jnts
            parent_skeleton(twod_jnt_mapping)

    #convert imageplane
    convert_images = False
    if convert_images:
        #set imageplane
        #convert openpose --write_images output to valid padding for maya
        for image_file in os.listdir(openpose_images):
            file_path = os.path.join(openpose_images, image_file)
            frame_idx = int(re.findall("(\d+)", image_file)[-1]) 
            os.rename(file_path, os.path.join(threed_pose_baseline, "maya/image_plane/image.{0}.png".format(frame_idx)))
        print "use", os.path.join(threed_pose_baseline, "maya/image_plane/"), " for imageplane."
