python main.py --imgroot ./images/demo1 \
--imglist ./splits/demo1.txt \
--saveroot ./images/demo1/results \
--verbose True \
--warping_progress True \
--resize 500 400 \
--ransac_max 500 \
--ransac_thres 10 \
--ransac_sample 6 \
--optimal_ransac True \
--ransac_inlier_prob 0.5 \
--sample_inlier 0.995 \
--gamma 0.0015 \
--sigma 8.5