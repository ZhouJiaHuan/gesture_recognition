import numpy as np
import dlib


def face_detect(detector, img_array, times=1, score_thr=-1):
    dets, scores, idx = detector.run(img_array, times, score_thr)
    return dets, scores, idx


def face_feature(face_pre, face_rec, img_array, det):
    shape = face_pre(img_array, det)
    face_chip = dlib.get_face_chip(img_array, shape)
    face_vector = face_rec.compute_face_descriptor(face_chip)
    return face_vector, face_chip


def faces_feature(face_pre, face_rec, img_array, dets):
    face_vectors = []
    for i, d in enumerate(dets):
        face_vectors.append(face_feature(face_pre, face_rec, img_array, d)[0])
    return face_vectors


def euclidean_dis(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2), ord=2)
