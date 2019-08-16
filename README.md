# motion_statistics
PyTorch implementation of self-supervised learning with motion statistics labels 

## Notes
0. Pretrain and finetune shared same weight initialization on final FC layer with std=0.01. (Changed, not much improvement on acc)
1. Separation on pattern three has no overlapping points (Changed, not much improvement on acc)
2. Weight decay is applied to both weights and biases (Changed, effect TBD)
3. During pre-train:
- num of epoch = 18 
- step size for learning rate decay is 6 
4. Label is output for each video clip (16 frames):
max_vl_p1, max_vo_p1, max_ul_p1, max_uo_p1, max_vl_p2, max_vo_p2, max_ul_p2, max_uo_p2, max_vl_p3, max_vo_p3, max_ul_p3, max_uo_p3, max(Mv), max(Mu)

