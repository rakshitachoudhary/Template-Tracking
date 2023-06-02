
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-p', '--pred_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-g', '--gt_path', type=str, default='groundtruth', required=True, \
                                                        help="Path for the ground truth masks folder")
    args = parser.parse_args()
    return args




def main(args):
    
    groundtruth=open(args.gt_path,'r')
    
    result=open(args.pred_path,'r')
    n=1
    mIoU=0
    while(True):
        ground_line=groundtruth.readline()
        result_line=result.readline()
        if not result_line:
            break
        
        ground_words=ground_line.split(",")
        x_ground_left=int(ground_words[0])
        x_ground_right=x_ground_left+int(ground_words[2])
        y_ground_bottom=int(ground_words[1])
        y_ground_top=y_ground_bottom+int(ground_words[3])
        
        result_words=result_line.split(',')
        x_result_left=int(result_words[0])
        x_result_right=x_result_left+int(result_words[2])
        y_result_bottom=int(result_words[1])
        y_result_top=y_result_bottom+int(result_words[3])
        
        #intersection area
        x_left=max(x_ground_left,x_result_left)
        x_right=min(x_ground_right,x_result_right)
        y_top=min(y_ground_top,y_result_top)
        y_bottom=max(y_ground_bottom,y_result_bottom)
        
        IoU=0
        if x_left>x_right or y_bottom>y_top:
            IoU=0
        else:
            intersection_area = (x_right-x_left+1)*(y_bottom-y_top+1)
            ground_area=(x_ground_right-x_ground_left+1)*(y_ground_bottom-y_ground_top+1)
            result_area=(x_result_right-x_result_left+1)*(y_result_bottom-y_result_top+1)
            union_area = ground_area+result_area-intersection_area
            
            IoU=(float(intersection_area))/union_area
        
        
        
        
        mIoU=((n-1)*mIoU+IoU)/n
        n+=1
       
        
    print("Mean IoU score is: ",mIoU)



if __name__ == "__main__":
    args = parse_args()
    main(args)