import os
from imageio import mimsave
import numpy as np
import torch
import torchvision.transforms.functional as F
import warnings
import matplotlib.pyplot as plt
import matplotlib
import cv2
from slowfast.utils import box_ops
import pickle
matplotlib.use('Agg')

def rescale_img(img):
    img = img - img.min()
    img = img / img.max() # [0,1]
    img = img * 255. # [0, 255]
    return img

def save_video_debug(imgs, path, name=''):
    os.makedirs(path, exist_ok=True)
    for i in range(len(imgs)):
        save_one_video(imgs[i], os.path.join(path, f"{i}_{name}.gif"))
        # mimsave(os.path.join(path, f"{i}_gt.gif"), list(imgs[i].cpu().numpy().transpose([1, 2, 3, 0])))

def save_one_video(imgs, path):
    imgs = imgs.cpu().numpy().transpose([1, 2, 3, 0])
    ret = []
    for img in imgs:
        if img.min() != 0. or img.max() != 255.:
            img = rescale_img(img)
        ret.append(img.astype(np.uint8))
    mimsave(path, ret)

def save_videos_with_boxes(imgs, allboxes, base, btype = 'cxcyhw', box_scores=None, box_names = None, normalized_boxes = True, verbose=False):
    # imgs: [BS, C, T, H, W]
    # boxes: [BS, T, O, 4]
    # box_scores: [BS,T, O]
    if imgs.dim() == 4: imgs = imgs.unsqueeze(0)
    if allboxes.dim() == 3: allboxes = allboxes.unsqueeze(0)
    BS = imgs.size(0)
    for b in range(BS):
        if verbose:
            print(f"SAVING VIDEO {b}/{BS}")
        path = os.path.join(base, f'vid_{b}')
        os.makedirs(path, exist_ok=True, mode=0o777)
        boxes = allboxes[b]
        boxes = boxes.reshape(boxes.size(0), -1 ,4)
        frames = imgs[b].detach().cpu().permute(1,2,3,0)
        draw_boxes_video(frames, boxes, path, btype = btype, 
                         box_scores=box_scores[b] if box_scores is not None else None,
                         box_names = box_names[b] if box_names is not None else None,
                         normalized_boxes = normalized_boxes)
    print("DONE")

def get_box_color(idx):
    l = [255,255,255]
    if idx < 3:
        l = [0,0,0]
        l[idx] = 255
    elif idx < 6:
        l = [255,255,255]
        l[idx - 3] = 0
    return tuple(l)

def draw_boxes_video(frames, boxes, save_path, btype = 'cxcyhw', box_scores = None, box_names = None, normalized_boxes = True):
    all = []
    for i in range(len(frames)):
        img = np.array(rescale_img(frames[i])).astype(np.float32)
        RGB_img_i =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        H_grid_size_i = RGB_img_i.shape[0] - 1
        W_grid_size_i = RGB_img_i.shape[1] - 1
        if btype in ['cxcyhw', 'cxcywh']: boxs = box_ops.box_cxcywh_to_xyxy(boxes[i])
        else: boxs = boxes[i]
        for ibox, box in enumerate(boxs):
            box_ = box.cpu().detach().numpy()
            if normalized_boxes:
                box_ = (box_ * np.array([W_grid_size_i, H_grid_size_i, W_grid_size_i, H_grid_size_i])).astype('int')
            else:
                box_ = box_.astype('int')
            top_left, bottom_right = box_[:2].tolist(), box_[2:].tolist()
            cv2.rectangle(RGB_img_i, tuple(top_left), tuple(bottom_right), get_box_color(ibox), 1)
            
            if box_scores is not None:
                x,y = tuple(top_left)
                score = str(round(box_scores[i,ibox].cpu().item(), 2))
                cv2.putText(RGB_img_i, score, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_box_color(ibox), 2)

            if tuple(top_left) != tuple(bottom_right):
                x,y = tuple(top_left)
                name = f'box_{ibox}'
                if box_names is not None:
                    try:
                        name = str(box_names[ibox])
                    except:
                        pass
                cv2.putText(RGB_img_i, name, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, get_box_color(ibox), 2)

        cv2.imwrite("{}/frame_{}.jpg".format(save_path, i), RGB_img_i)
        RGB_img_i =  cv2.cvtColor(RGB_img_i, cv2.COLOR_BGR2RGB).astype(np.uint8)
        all.append(RGB_img_i)
    mimsave("{}/vid.gif".format(save_path), all)



def save_attn(attn, path):
    os.makedirs(path, exist_ok=True)
    for i in range(len(imgs)):
        mimsave(os.path.join(path, f"{i}_gt.gif"), list(imgs[i].cpu().numpy().transpose([1, 2, 3, 0])))

def reshape_attn(imgs, attn):
    imgs = imgs.detach().cpu()
    attn = attn.detach().cpu()
    BS, C, T, H, W = imgs.shape
    n_slots = 4
    Tattn = T//2
    Wattn = Hattn = int((attn.size(-1))**0.5)
    attn = attn.reshape(BS,T, n_slots, Hattn, Wattn).permute(0,2,1,3,4)
    return attn


def save_attn_on_img(args, imgs, attn, bpath):
    imgs = imgs.detach().cpu()
    attn = attn.detach().cpu()
    BS, C, T, H, W = imgs.shape
    n_slots = attn.size(1)
    Tattn = T
    Wattn = Hattn = int((attn.size(-1) // Tattn)**0.5)
    attn = attn[:BS]
    attn = attn.reshape(BS, n_slots, Tattn, Hattn, Wattn)
    attn = attn.reshape(-1, Tattn, Hattn, Wattn)
    attn = F.resize(attn, size=(H,W))
    # attn = attn.reshape(BS, n_slots, T//2, H, W)
    # attn = attn.reshape(BS, n_slots, T//2, 1,H,W).expand(BS, n_slots, T//2, 2,H,W).reshape(BS, n_slots, T,H,W)
    for b in range(BS):
        bimgs = imgs[b].reshape(-1, C,T,H,W).expand(n_slots,C,T,H,W).permute(1,0,2,3,4) # [C,n_slots,T,H,W]
        comb = bimgs * attn[b,...]
        comb = comb.permute(1,0,2,3,4) # [n_slots, C,T,H,W]
        path = os.path.join(bpath, str(b), 'attn_on_img')
        os.makedirs(path, exist_ok=True, mode=0o777)
        save_video_debug(comb, path, name = 'slot')


def save_attn_on_img_OT(args, imgs, attn, bpath):
    imgs = imgs.detach().cpu()
    attn = attn.detach().cpu()
    BS, C, T, H, W = imgs.shape
    n_slots = attn.size(1)
    O = args.num_queries
    assert n_slots == O * T

    Tattn = T
    Wattn = Hattn = int((attn.size(-1) // Tattn)**0.5)


    attn = attn[:BS]
    attn = attn.reshape(BS, T, O, Tattn, Hattn, Wattn)
    attn = attn_reshaped =  attn.mean(1) # [BS,O, Tattn, Hattn, Wattn]
    attn = attn.reshape(-1, Tattn, Hattn, Wattn)
    attn = F.resize(attn, size=(H,W))
    attn = attn.reshape(BS ,O, Tattn, H, W)
    # attn = attn.reshape(BS, n_slots, T//2, H, W)
    # attn = attn.reshape(BS, n_slots, T//2, 1,H,W).expand(BS, n_slots, T//2, 2,H,W).reshape(BS, n_slots, T,H,W)
    for b in range(BS):
        bimgs = imgs[b].reshape(1, C,T,H,W).expand(O,C,T,H,W).permute(1,0,2,3,4) # [C,n_slots,T,H,W]
        comb = bimgs * attn[b,...]
        comb = comb.permute(1,0,2,3,4) # [n_slots, C,T,H,W]
        path = os.path.join(bpath, str(b), 'attn_on_img')
        os.makedirs(path, exist_ok=True, mode=0o777)
        save_video_debug(comb, path, name = 'slot')
    return attn_reshaped


def plot_attn_batch(imgs, attn, bpath):
    imgs = imgs.detach().cpu()
    attn = attn.detach().cpu()
    attn = reshape_attn(imgs, attn) # [bs, slots, T ,H ,W]
    os.makedirs(bpath, exist_ok = True, mode=0o777)
    for i in range(attn.size(0)):
        plot_attn(attn[i], os.path.join(bpath, f'{i}_attn.jpg'))

def plot_attn(attn, path):
    S, T, H, W = attn.shape
    plt.figure(figsize=(3*T,2.5*S))
    for s in range(S):
        for t in range(T):
            plt.subplot(S, T, s*T + t + 1)
            plt.imshow(attn[s,t])
    plt.savefig(path)



def plot_attn_batch2(imgs, attn, bpath):
    imgs = imgs.detach().cpu()
    attn = attn.detach().cpu() # [BS, O, T , H, W]
    os.makedirs(bpath, exist_ok = True, mode=0o777)
    for i in range(attn.size(0)):
        plot_attn2(imgs[i], attn[i], os.path.join(bpath, f'{i}_attn.jpg'))


def plot_attn2(imgs, attn, path):
    S, T, H, W = attn.shape
    Timg = imgs.shape[0]
    step = 2
    Ttot = T // step
    plt.figure(figsize=(3*Ttot,4.5*S))

    for s in range(S + 1):
        for t in range(Ttot):
            plt.subplot((S + 1), Ttot, s*Ttot + t + 1)
            if s == 0: # plot image
                img = imgs[:,t * step].permute(1,2,0).cpu().numpy()
                img = img - img.min()
                img = img / img.max()
                plt.imshow(img, vmin=0, vmax=1)
            else:
                plt.imshow(attn[s-1,t * step], vmin=0, vmax=1)
    plt.savefig(path)
    plt.close()

###

def AG_tuple_to_txt(args, tup):
    # (x,y,z,w)
    names = ['class', 'relation']
    ret = []
    for i, name in enumerate(names):
        idx = tup[i]
        ret.append(args.vocab[name]['idx_to_name'].get(str(idx), 'None'))
    return tuple(ret)

def AG_tuples_to_txt(args, tups):
    tups = list(map(lambda x: AG_tuple_to_txt(args, x), tups))
    return tups

def save_text_AG(args, targets, preds, base, threshold = 0.5):
    names = ['class', 'relation']
    rel_name = names[-1]
    BS = args.batch_size_val
    class_preds = torch.argmax(preds['pred_logits']['class'], dim = -1) # [BS, O]
    rel_preds = torch.sigmoid(preds['pred_logits']['relation'])

    target = {n:torch.stack([t['label_boxes'][n] for t in targets]) for n in names} # [BS]
    txt = ""
    for b in range(BS):
        ltxt = f"############ Batch {b}:" + "\n"
        for o in range(args.num_queries):
            ltxt = f"###### Slot {o}:" + "\n"
            if target['class'][b, o] == args.vocab['class']['n']: continue
            class_pred = args.vocab['class']['idx_to_name'].get(str(class_preds[b,o].item()), 'none')
            class_true = args.vocab['class']['idx_to_name'].get(str(target['class'][b, o].item()), 'none') 


            npreds = [args.vocab[rel_name]['idx_to_name'].get(str(i), 'none') for i,p in enumerate(rel_preds[b, o]) if p > threshold]
            ntrue = [args.vocab[rel_name]['idx_to_name'].get(str(i), 'none') for i,p in enumerate(target[rel_name][b,o]) if p > threshold]

            npreds = list(set(npreds) - set(['none']))
            ntrue = list(set(ntrue) - set(['none']))

            ltxt = ltxt + f"predicted class: {class_pred}" + '\n'
            ltxt = ltxt + f"target class: {class_true}" + '\n'
            ltxt = ltxt + f"predicted: {npreds}" + '\n'
            ltxt = ltxt + f"target: {ntrue}" + '\n'

            ltxt = ltxt + f"target_tuples: {AG_tuples_to_txt(args, targets[b]['tuples'])}" + '\n'
            txt = txt + ltxt + '\n'
    with open(os.path.join(base, 'labels_preds.txt'), 'wt') as f:
        f.write(txt)

def save_text(args, targets, preds, base):
    if args.dataset == 'AG':
        save_text_AG(args, targets, preds, base)
        return
    if args.dataset not in verb2name: return        
    preds = preds['pred_logits'].argmax(-1) # [BS]
    target = torch.cat([t['labels'] for t in targets]) # [BS]
    f = open(os.path.join(base, 'labels_preds.txt'), 'wt')
    idx2v = verb2name['something']
    for b in range(preds.size(0)):
        p, t = preds[b].item(), target[b].item()
        f.write(f"batch_idx: {b}, video_name: {targets[b]['name']}, label: {t, idx2v[t]}, pred: {p, idx2v[p]}" + '\n')
    f.close()

def get_ssv2_dict():
    import json
    path = "/home/gamir/DER-Roei/datasets/smthsmth/sm/annotations/something-something-v2-labels.json"
    with open(path, 'rt') as f:
        d = json.load(f)
    ret = {int(v):k for k,v in d.items()}
    return ret

def get_AG_dict():
    names = ['class', '']
    return {}
try:
    verb2name = {'something':get_ssv2_dict()}
except:
    pass


##

def visualize_batch(args,eval_dir,i_batch,imgs, box_tensors, box_categories, target ,out, loss):
    if args.dataset == 'AGQA': return 
    eval_dir_img = os.path.join(eval_dir, f'{i_batch}')
    os.makedirs(eval_dir_img, exist_ok=True, mode=0o777)
    vid = torch.stack([v[1].tensors for v in imgs])

    save_dict = \
        {
            'imgs': vid,
            'preds': out,
            'annotation': target,
            'box_tensors':box_tensors,
            'box_categories':box_categories,
            'met_lst': None,
            'loss_lst': loss,
            'vocab':getattr(args, 'vocab', None)
        }
    save_path = os.path.join(eval_dir_img, "d.pkl")
    with open(save_path , 'wb') as f:
        pickle.dump(save_dict, f)
    # torch.save(save_dict,save_path)
    # if out is not None: save_text(args, target, out, eval_dir_img)
    if 'box_categories_names' in target[0]:
        box_names = [t['box_categories_names'] for t in target]
    else:
        box_names = None

    if isinstance(box_tensors, (tuple, torch.Tensor, list)) and len(box_tensors) > 0 and utils.safe_len(box_tensors[0]) > 0:
        save_videos_with_boxes(vid, box_tensors, eval_dir_img, box_names = box_names)
    else:
        save_video_debug(vid, eval_dir_img, name= 'vid')
    
    if out is not None and 'boxes' in out:
        save_videos_with_boxes(vid, out['boxes'], os.path.join(eval_dir_img, 'pred_boxes_vids'), box_scores=out.get('boxes_categories', None))

    if out is not None and 'attn' in out:
        attn = out.get('attn_unscaled', out['attn'])[:len(vid)]
        attn_reshaped = save_attn_on_img_OT(args, vid, attn, eval_dir_img)
        plot_attn_batch2(vid, attn_reshaped, eval_dir_img)
