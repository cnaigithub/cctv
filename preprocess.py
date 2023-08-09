import os
import cv2
from tqdm import tqdm


def xywh2yolo(size,x,y,w,h):
    dh = 1./size[0]
    dw = 1./size[1]
    return (x*dw, y*dh, w*dw, h*dh)

excs = []
output = '/mnt/cn_hdd/video_output'
rts = ['/mnt/cn_hdd/video2/','/mnt/cn_hdd/video/']
for rt in rts:
    vsize_dic = {}
    #vid2frame
    # rt = '/mnt/cn_hdd/video/'
    print('extracting frames . . .')
    folders = [i for i in os.listdir(rt) if not i.startswith('.')]
    for folder in tqdm(folders):
        if folder == 'Aregion-12-3':
            continue
        files = [i for i in os.listdir(os.path.join(rt,folder)) if not i.startswith('.')]
        # breakpoint()

        for file in tqdm(files):
            try:        
                print(f'processing {file}')
                fname = file.split('/')[-1].split('.')[0]
                #vid2frame
                if '.mp4' in file:

                    vid = cv2.VideoCapture(os.path.join(rt,folder,file))
                    count =0
                    while True:
                        ret, frame = vid.read()
                        if count==0:
                            vsize_dic[fname] = frame.shape[:2]
                        if not ret:
                            break
                        cv2.imwrite(f'{output}/{fname}_{count}.jpg' ,frame)
                        count +=1
                    vid.release()
                    cv2.destroyAllWindows()
            except:
                excs.append(file)
                breakpoint()

    print('converting labels . . .')
    # label2yolo
    frame_dic = {}
    folders = [i for i in os.listdir(rt) if not i.startswith('.')]
    for folder in tqdm(folders):
        #loop for a file
        files = [i for i in os.listdir(os.path.join(rt,folder)) if not i.startswith('.')]
        for file in tqdm(files):        
            try:
                print(f'processing {file}')
                #label2yolo
                if '.txt' in file:
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
    breakpoint()