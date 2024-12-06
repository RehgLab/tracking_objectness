def mask_loss_parallel(masks, traj_g, traj_pred, valids):

    B,S,N,_ = traj_g.shape
    device = traj_g.device
    traj_g_loss = traj_g.clone()
    traj_pred_loss = traj_pred.clone()
    B,S,H,W = masks.shape
    valids_loss = valids.clone()
    for b in range(B):
        for si in range(S):
            oob_inds_g = torch.logical_or(
                torch.logical_or(traj_g[b,si,:,0]<0, traj_g[b,si,:,0]>W-1),
                torch.logical_or(traj_g[b,si,:,1]<0, traj_g[b,si,:,1]>H-1))
            #print(torch.logical_or(traj_g[b,si,:,0]<0, traj_g[b,si,:,0]>W-1).shape)
            #print(oob_inds_g.shape)
            oob_inds_pred = torch.logical_or(
                torch.logical_or(traj_pred[b,si,:,0]<0, traj_pred[b,si,:,0]>W-1),
                torch.logical_or(traj_pred[b,si,:,1]<0, traj_pred[b,si,:,1]>H-1))
            oobs_inds = torch.logical_or(oob_inds_g, oob_inds_pred)
            #print(oob_inds_pred.shape)
            #print(oobs_inds.shape)
            valids_loss[b,si, oobs_inds] = 0
    traj_g_loss[:,:,:,0]= (traj_g[:,:,:,0].clone()*valids_loss).int()
    traj_g_loss[:,:,:,1] = (traj_g[:,:,:,1].clone()*valids_loss).int()
    traj_pred_loss[:,:,:,0] = (traj_pred[:,:,:,0].clone()*valids_loss).int()
    traj_pred_loss[:,:,:,1] = (traj_pred[:,:,:,1].clone()*valids_loss).int()
    mask_matches = torch.zeros((B,S,N), dtype = torch.float64).to(device)
    count = 0
    for b in range(B):
        for si in range(S):
            mask_g = masks[b,si,traj_g_loss[b,si,:,1].int(),traj_g_loss[b,si,:,0].int()]
            mask_pred = masks[b,si,traj_pred_loss[b,si,:,1].int(),traj_pred_loss[b,si,:,0].int()]
            #count+=np.sum( (mask_g!=mask_pred))
            mask_matches[b,si,:] = (mask_g==mask_pred)

    #print(count)
    #return torch.from_numpy(mask_matches).to(device)
    return mask_matches

def mask_distance_loss(trajs_g, pred, mask):

    trajs_g = torch.nan_to_num(trajs_g)
    pred = torch.nan_to_num(pred)
    #print("mask sum", torch.sum(mask))
    i_loss = torch.mean((trajs_g-pred).abs(), dim=3)
    loss = utils.basic.reduce_masked_mean(i_loss, mask)

    return loss


#calculate mask loss
mask_matches_p = mask_loss_parallel(masks, trajs_g, coords*self.stride, valids)

matching_valids_p = (1-mask_matches_p)*valids
print("matching valids sum", torch.sum(matching_valids_p))
masked_loss = mask_distance_loss(trajs_g, coords*self.stride, matching_valids_p)


#final loss
loss = sequence_loss(coord_predictions1, trajs_g, vis_g, torch.nan_to_num(valids), 0.8) + 0.1*masked_loss

