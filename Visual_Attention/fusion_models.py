import torch
import torch.nn as nn
import torch.nn.functional as F
class mfh_baseline(nn.Module):
    def __init__(self,QUEST_EMBED,VIS_EMBED,MFB_FACTOR_NUM=5,MFB_OUT_DIM=1000,MFB_DROPOUT_RATIO=0.1,NUM_OUTPUT_UNITS=2):
        super(mfh_baseline, self).__init__()
        
        self.JOINT_EMB_SIZE = MFB_FACTOR_NUM * MFB_OUT_DIM
        self.MFB_OUT_DIM=MFB_OUT_DIM
        self.MFB_FACTOR_NUM=MFB_FACTOR_NUM
        self.Linear_dataproj1 = nn.Linear(QUEST_EMBED, self.JOINT_EMB_SIZE)
        self.Linear_dataproj2 = nn.Linear(QUEST_EMBED, self.JOINT_EMB_SIZE)
        self.Linear_imgproj1 = nn.Linear(VIS_EMBED, self.JOINT_EMB_SIZE)
        self.Linear_imgproj2 = nn.Linear(VIS_EMBED, self.JOINT_EMB_SIZE)
        #self.Linear_predict = nn.Linear(MFB_OUT_DIM * 2, NUM_OUTPUT_UNITS)
        #self.Dropout1 = nn.Dropout(p=opt.LSTM_DROPOUT_RATIO)
        #self.Dropout2 = nn.Dropout(MFB_DROPOUT_RATIO)

    def forward(self, q_feat, img_feat):
        
        mfb_q_o2_proj = self.Linear_dataproj1(q_feat)                       # data_out (N, 5000)
        mfb_i_o2_proj = self.Linear_imgproj1(img_feat.float())              # img_feature (N, 5000)
        mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)
        mfb_iq_o2_drop = mfb_iq_o2_eltwise
        #mfb_iq_o2_drop = self.Dropout2(mfb_iq_o2_eltwise)
        mfb_iq_o2_resh = mfb_iq_o2_drop.view(-1, 1, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)
        if(mfb_iq_o2_resh.size(0)>1):                                                                                             # N x 1 x 1000 x 5
            mfb_o2_out = torch.squeeze(torch.sum(mfb_iq_o2_resh, 3)) 
        else:
            mfb_o2_out = torch.sum(mfb_iq_o2_resh, 3).view(1,mfb_iq_o2_resh.size(2))                    # N x 1000
        mfb_o2_out = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))
        #print(mfb_o2_out.size())       # signed sqrt
        mfb_o2_out = F.normalize(mfb_o2_out)
        

        mfb_q_o3_proj = self.Linear_dataproj2(q_feat)                   # data_out (N, 5000)
        mfb_i_o3_proj = self.Linear_imgproj2(img_feat.float())          # img_feature (N, 5000)
        mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)
        mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_iq_o2_drop)
        mfb_iq_o3_drop = mfb_iq_o3_eltwise
        #mfb_iq_o3_drop = self.Dropout2(mfb_iq_o3_eltwise)
        mfb_iq_o3_resh = mfb_iq_o3_drop.view(-1, 1, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)

        #mfb_o3_out = torch.squeeze(torch.sum(mfb_iq_o3_resh, 3))                            # N x 1000
        if(mfb_iq_o3_resh.size(0)>1):                                                                                             # N x 1 x 1000 x 5
            mfb_o3_out = torch.squeeze(torch.sum(mfb_iq_o3_resh, 3)) 
        else:
            mfb_o3_out = torch.sum(mfb_iq_o3_resh, 3).view(1,mfb_iq_o3_resh.size(2)) 
        mfb_o3_out = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
        mfb_o3_out = F.normalize(mfb_o3_out)

        mfb_o23_out = torch.cat((mfb_o2_out, mfb_o3_out), 1)#200,2000     
        #prediction = self.Linear_predict(mfb_o23_out)               
        #prediction = F.log_softmax(prediction)

        return mfb_o23_out