import os
from tqdm import tqdm
import cv2

def xywh2yolo(size,x,y,w,h):
    dh = 1./size[0]
    dw = 1./size[1]
    return (x*dw, y*dh, w*dw, h*dh)

excs = ['A-12-3-(41).txt', 'A-12-3-(35).txt', 'A-12-3-(6).txt', 'A-12-3-(10).txt', 'A-12-3-(17).txt', 'A-12-3-(57).txt', 'A-12-3-(74).txt', 'A-12-3-(22).txt', 'A-12-3-(37).txt', 'A-12-3-(49).txt', 'A-12-3-(21).txt', 'A-12-3-(42).txt', 'A-12-3-(28).txt', 'A-12-3-(31).txt', 'A-12-3-(5).txt', 'A-12-3-(68).txt', 'A-12-3-(15).txt', 'A-12-3-(20).txt', 'A-12-3-(30).txt', 'A-12-3-(9).txt', 'A-12-3-(8).txt', 'A-12-3-(18).txt', 'A-12-3-(60).txt', 'A-12-3-(25).txt', 'A-12-3-(47).txt', 'A-12-3-(29).txt', 'A-12-3-(43).txt', 'A-12-3-(65).txt', 'A-12-3-(32).txt', 'A-12-3-(27).txt', 'A-12-3-(52).txt', 'A-12-3-(70).txt', 'A-12-3-(56).txt', 'A-12-3-(72).txt', 'A-12-3-(58).txt', 'A-12-3-(2).txt', 'A-12-3-(14).txt', 'A-12-3-(59).txt', 'A-12-3-(24).txt', 'A-12-3-(67).txt', 'A-12-3-(45).txt', 'A-12-3-(54).txt', 'A-12-3-(23).txt', 'A-12-3-(63).txt', 'A-12-3-(3).txt', 'A-12-3-(55).txt', 'A-12-3-(62).txt', 'A-12-3-(61).txt', 'A-12-3-(12).txt', 'A-12-3-(39).txt', 'A-12-3-(36).txt', 'A-12-3-(1).txt', 'A-12-3-(13).txt', 'A-12-3-(38).txt', 'A-12-3-(53).txt', 'A-12-3-(33).txt', 'A-12-3-(11).txt', 'A-12-3-(19).txt', 'A-12-3-(66).txt', 'A-12-3-(16).txt', 'A-12-3-(46).txt', 'A-12-3-(4).txt', 'A-12-3-(73).txt']
rt = '/mnt/cn_hdd/video2'
output = '/mnt/cn_hdd/video_output'
folders = [i for i in os.listdir(rt) if not i.startswith('.')]

vsize_dic = {}
print('extracting vsize dic . . .')
folders = [i for i in os.listdir(rt) if not i.startswith('.')]
for folder in tqdm(folders):
    files = [i for i in os.listdir(os.path.join(rt,folder)) if not i.startswith('.')]
    for file in tqdm(files):
        try:        
            print(f'processing {file}')
            fname = file.split('/')[-1].split('.')[0]
            #vid2frame
            if '.mp4' in file:
                vid = cv2.VideoCapture(os.path.join(rt,folder,file))
                ret, frame = vid.read()
                vsize_dic[fname] = frame.shape[:2] 
                vid.release()
                cv2.destroyAllWindows()
        except:
            excs.append(file)
breakpoint()
frame_dic = {}
for folder in tqdm(folders):
    files = [i for i in os.listdir(os.path.join(rt,folder)) if not i.startswith('.')]
    # breakpoint()
    for file in tqdm(files):
        try:
            print(f'processing {file}')
            #label2yolo
            if '.txt' in file and file in excs:
                fname = file.split('/')[-1].split('.')[0]
                vsize = vsize_dic[fname]
                f = open(os.path.join(rt,folder,file), 'r')
                lines = [l.strip() for l in f.readlines()]
                f.close()
                for line in lines:
                    split_line = line.split(',')
                    if split_line[0] not in frame_dic.keys():
                        frame_dic[split_line[0]] = []
                    frame_dic[split_line[0]].append([int(x) for x in split_line[3:]])
                # breakpoint()
                for f_num, labels in frame_dic.items():
                    new_f = open(f'{output}/{fname}_{f_num}.txt', 'w')
                    for label in labels:
                        ret = xywh2yolo(vsize,label[0],label[1],label[2],label[3])
                        new_f.write(f'0 {ret[0]} {ret[1]} {ret[2]} {ret[3]}\n')
                    new_f.close()
        except:
            excs.append(file)
            breakpoint()
