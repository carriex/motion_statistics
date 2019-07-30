# motion_statistics
PyTorch implementation of self-supervised learning with motion statistics labels 

## Notes
0. No normalization is done on u_flow and v_flow value
1. Separation on pattern three has no overlapping points 
2. During pre-train:
- num of epoch = 18 
- step size for learning rate decay is 6 
3. Label is output for each video clip (16 frames):
max_vl_p1, max_vo_p1, max_ul_p1, max_uo_p1, max_vl_p2, max_vo_p2, max_ul_p2, max_uo_p2, max_vl_p3, max_vo_p3, max_ul_p3, max_uo_p3, max(Mv), max(Mu)

