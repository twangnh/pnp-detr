class SortSampler(nn.Module):

    def __init__(self, topk_ratio, input_dim, score_pred_net='2layer-fc-256', kproj_net='1layer-fc', unsample_abstract_number=30,pos_embed_kproj=False):
        ##topk_ratio : hard sample的比例
        ## unsample_abstract_number： soft sample的数量，是一个固定值
        super().__init__()
        self.topk_ratio = topk_ratio
        if score_pred_net == '2layer-fc-256':
            self.score_pred_net = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(input_dim, 1))
        elif score_pred_net == '2layer-fc-16':
            self.score_pred_net = nn.Sequential(nn.Linear(input_dim, 16),
                                                 nn.ReLU(),
                                                 nn.Linear(16, 1))
        elif score_pred_net == '1layer-fc':
            self.score_pred_net = nn.Linear(input_dim, 1)
        else:
            raise ValueError

        self.norm_feature = nn.LayerNorm(input_dim,elementwise_affine=False)
        self.unsample_abstract_number = unsample_abstract_number
        if kproj_net == '2layer-fc':
            self.k_proj = nn.Sequential(nn.Linear(input_dim, input_dim),
                                                nn.ReLU(),
                                                nn.Linear(input_dim, unsample_abstract_number))
        elif kproj_net == '1layer-fc':
            self.k_proj = nn.Linear(input_dim, unsample_abstract_number)
        else:
            raise ValueError
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed_kproj = pos_embed_kproj

    def forward(self, src, mask, pos_embed):
        #pos_embed shape: h*w, 1, c
        l, bs ,c  = src.shape
        if mask==None:
            mask = src.new_zeros(bs,l).bool()
            pos_embed = pos_embed.repeat(1,bs,1)
        sample_weight = self.score_pred_net(src).sigmoid().view(bs,-1)
        # sample_weight[mask] = sample_weight[mask].clone() * 0.
        # sample_weight.data[mask] = 0.
        sample_weight_clone = sample_weight.clone().detach()
        sample_weight_clone[mask] = -1.

        ##max sample number:
        sample_lens = ((~mask).sum(1)*self.topk_ratio).int()
        max_sample_num = sample_lens.max()
        mask_topk = torch.arange(max_sample_num).expand(len(sample_lens), max_sample_num).to(sample_lens.device) > (sample_lens-1).unsqueeze(1)

        ## for sampling remaining unsampled points
        min_sample_num = sample_lens.min()

        sort_order = sample_weight_clone.sort(descending=True,dim=1)[1]
        sort_confidence_topk = sort_order[:,:max_sample_num]
        sort_confidence_topk_remaining = sort_order[:,min_sample_num:]
        ## flatten for gathering
        src = src.flatten(2).permute(2, 0, 1)
        src = self.norm_feature(src)

        src_sample_remaining = src.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))

        ## this will maskout the padding and sampled points
        mask_unsampled = torch.arange(mask.size(1)).expand(len(sample_lens), mask.size(1)).to(sample_lens.device) < (sample_lens).unsqueeze(1)
        mask_unsampled = mask_unsampled | mask.gather(1, sort_order)
        mask_unsampled = mask_unsampled[:,min_sample_num:]

        ## abstract the unsampled points with attention
        if self.pos_embed_kproj:
            pos_embed_sample_remaining = pos_embed.gather(0, sort_confidence_topk_remaining.permute(1, 0)[..., None].expand(-1, -1, c))
            kproj = self.k_proj(src_sample_remaining+pos_embed_sample_remaining)
        else:
            kproj = self.k_proj(src_sample_remaining)
        kproj = kproj.masked_fill(
            mask_unsampled.permute(1,0).unsqueeze(2),
            float('-inf'),
        ).permute(1,2,0).softmax(-1)
        abs_unsampled_points = torch.bmm(kproj, self.v_proj(src_sample_remaining).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_pos_embed = torch.bmm(kproj, pos_embed.gather(0,sort_confidence_topk_remaining.
                                                                          permute(1,0)[...,None].expand(-1,-1,c)).permute(1,0,2)).permute(1,0,2)
        abs_unsampled_mask = mask.new_zeros(mask.size(0),abs_unsampled_points.size(0))

        ## reg sample weight to be sparse with l1 loss
        sample_reg_loss = sample_weight.gather(1,sort_confidence_topk).mean()
        src_sampled = src.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c)) *sample_weight.gather(1,sort_confidence_topk).permute(1,0).unsqueeze(-1)
        pos_embed_sampled = pos_embed.gather(0,sort_confidence_topk.permute(1,0)[...,None].expand(-1,-1,c))
        mask_sampled = mask_topk

        src = torch.cat([src_sampled, abs_unsampled_points])
        pos_embed = torch.cat([pos_embed_sampled,abs_unsampled_pos_embed])
        mask = torch.cat([mask_sampled, abs_unsampled_mask],dim=1)
        assert ((~mask).sum(1)==sample_lens+self.unsample_abstract_number).all()
        return src, sample_reg_loss, sort_confidence_topk, mask, pos_embed