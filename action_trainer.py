import torch
import torch.utils.data 

import numpy as np
from opt import opt
from video_loader import ImageLoader, DetectionLoader2, DetectionProcessor2, DataWriter2, Mscoco
from webcam_loader import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, crop_from_dets, Mscoco
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *
import os
import sys
from tqdm import tqdm
import time
import cv2
from fn import *
from pPose_nms import write_json
import json
args = opt
args.dataset = 'coco'
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
def loop():
    n = 0
    while True:
        yield n
        n += 1

def get_pose(i, Cap, flage=False):
    
        Tnamefile = get_frame(i, Cap)        
        t, Pose = get_joints(Tnamefile, i)
        Old_pose = Pose['result']
        Old_pose = np.array(Old_pose[0]['keypoints'])
        New_pose = old2new_joint(Old_pose) 
        if flage:
            return  Tnamefile,  New_pose, Pose
        else:
            return  Tnamefile,  New_pose
def get_joints(Frame1, i):
 
    img, orig_img, im_dim_list= ImageLoader.cnv_img(Frame1)
    orig_img,boxes,scores,inps,pt1,pt2 = Dloader.load(img, orig_img, im_dim_list)
    orig_img,boxes,scores,inps,pt1,pt2 = DetectionProcessor2.process(orig_img,boxes,scores,inps,pt1,pt2)
    batchSize = 80 #args.posebatch
    with torch.no_grad():

        hm = []
        j=0
        inps_j = inps[j*batchSize:min((j +  1)*batchSize, 1)].cuda(torchCuda)
        hm_j = pose_model(inps_j)
        hm.append(hm_j)
        hm = torch.cat(hm)
        hm = hm.cpu()
        img,joints = DataWriter2.update(boxes, scores, hm, pt1, pt2, orig_img , i)
    return img,joints

start= time.time()
start1= time.time()
if __name__ == "__main__":
#    try:
        
        start_time = time.time()
        webcam = args.webcam
        videofile = args.T_path
        pose_dataset = Mscoco()
        if args.fast_inference:
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)
            
        pose_model.cuda()
        pose_model.eval()
        Cap_T = cv2.VideoCapture(videofile)
        img = get_frame(1,Cap_T)
        total_len = count_frames_manual(Cap_T)
        poses_T_1 = np.zeros(shape=(total_len,15,2))
        poses_T_2 = np.zeros(shape=(total_len,30))
        pose_id = np.zeros(shape=(total_len), dtype = int)
        
        torchCuda=0
        Dloader=DetectionLoader2()
#        mode = args.mode
        if not os.path.exists(args.outputpath):
            os.mkdir(args.outputpath)
        if not len(videofile):
            raise IOError('Error: must contain --video')
 
        name_T = (videofile.split("/")[-1]).split(".")[0]
        
        if not os.path.exists(os.path.join(args.outputpath, name_T)):
    
            print("Warning: Your video is being processed for the first time and might take a few minutes ...")
            os.mkdir(os.path.join(args.outputpath, name_T))
            final_result =[]
            j=0
            k=0
            for k in tqdm(range(total_len)):
                    
                    try:
                        img, poses_T_1[k], pose = get_pose(k, Cap_T, flage=True)                 
                        align_pose_T =  align_torso(poses_T_1[k])
                        poses_T_2[k] = change_dim(align_pose_T)                     
                        id_ = pose['imgname']
                        pose_id[k] = int(id_)
                        final_result.append(pose)
                        
                    except:
                            print("number==>",k)
                            poses_T_1[k]=0
                            poses_T_2[k]=0
                            pose_id[k]=0
#                            continue
                            
            write_json(final_result, os.path.join(args.outputpath, name_T))  
            print("Video process completed ...")
        elif not os.path.isfile(os.path.join(os.path.join(args.outputpath, name_T), "action_trainer_results.json")):
              
              
              print("Warning: Your video is being processed for the first time and might take a few minutes ...")
              final_result =[]
#              k=0
              for i in tqdm(range(total_len)):
                    try:
                        img, poses_T_1[i], pose = get_pose(i, Cap_T, flage=True)                 
                        align_pose_T =  align_torso(poses_T_1[i])
                        poses_T_2[i] = change_dim(align_pose_T)
                        k = pose['imgname']
                        pose_id[i] = int(k)
                        final_result.append(pose)
                        
                    except:
                             print("video_number==>",i)
                             poses_T_1[i]=0
                             poses_T_2[i]=0
                             pose_id[i]=0
                             
              write_json(final_result, os.path.join(args.outputpath, name_T))  
              print("Video process completed ...")
        else:
             print("An existing processed video has been found on your machine...")
             json_file_root = os.path.join(os.path.join(args.outputpath, name_T),"action_trainer_results.json")
             json_file = open(json_file_root, "r")
             json_string = json_file.readline()
             json_dict = json.loads(json_string)
#             i=0
             for pose  in json_dict:

                        k = pose['image_id']
                        k= int(k)
                        pose_id[k] = k
                        Old_pose = np.array(pose['keypoints'])
                        poses_T_1[k] = old2new_joint2(Old_pose)
                        align_pose_T =  align_torso(poses_T_1[k])
                        poses_T_2[k] = change_dim(align_pose_T)


        
        data_loader = WebcamLoader(webcam).start()
        (fourcc,fps,frameSize) = data_loader.videoinfo()
#        print("***************************framerate",fps)
#          Load detection loader
        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader(data_loader, batchSize=args.detbatch).start()
        det_processor = DetectionProcessor(det_loader).start()
        save_path = os.path.join(args.outputpath, 'Action_trainer_webcam'+webcam+'.avi')
        writer = DataWriter(Cap_T, poses_T_2, poses_T_1, pose_id, args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps,frameSize, total_len).start()
    
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }
    
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc =  tqdm(loop())
        batchSize = args.posebatch
        for i in im_names_desc:
            try:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                        continue
    
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                    # Pose Estimation
                    
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
    
                    hm = hm.cpu().data
                    writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
    
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)
                if args.profile:
                    # TQDM
                    im_names_desc.set_description(
                    'det time: {dt:.3f} | pose time: {pt:.2f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn'])))
            except KeyboardInterrupt:
                break
    
        print(' ')
        print('===========================> Finish Model Running.')
        if (args.save_img or args.save_video) and not args.vis_fast:
                print('===========================> Rendering remaining images in the queue...')
                print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
#        while(writer.running()):
#            pass
#        writer.stop()     
