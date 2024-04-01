import numpy as np


def angleHands(joint_hands):
    # 가시성 정보가 0인 경우 모든 값을 0으로 설정하여 반환
    if (joint_hands[:, 3] == 0).any():
        return np.zeros((15,))
    
    # 관절 간의 각도 계산
    v1 = joint_hands[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint  각 관절은 [x, y, z] 좌표로 표현되므로 :3
    v2 = joint_hands[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
    v = v2 - v1 # [20, 3]. 20개 행과 3개 열

    # 벡터 정규화
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # 각도 계산 (arccos를 이용하여 도트 곱의 역순 취함)
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
    angle = np.degrees(angle)

    # 라벨에 각도 정보 추가
    angle_label = np.array([angle], dtype=np.float32)
    
    return angle_label.flatten()


def anglePose(joint_pose):
    # 가시성 정보가 0인 경우 모든 값을 0으로 설정하여 반환
    if (joint_pose[:, 3] == 0).any():
        return np.zeros((15,))
    
    v1 = joint_pose[[0,0,0,0,7,8,9,10,11,12,11,12,11,12,0,0], :3]
    v2 = joint_pose[[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
    # 0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    # 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

    v = v2 - v1

    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],:], 
        v[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],:])) # [15,]
    angle = np.degrees(angle)

    angle_label = np.array([angle], dtype=np.float32)

    return angle_label.flatten()


def setVisibility(x, y, z):
    if x != 0 and y != 0 and z != 0:
        return 1
    elif x == 0 and y == 0 and z == 0:
        return 0
    else:
        return 0.5