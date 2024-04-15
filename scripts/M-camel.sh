python dynamicTypography.py --word "CAMEL" --optimized_letter "M" \
--caption "A camel walks steadily across the desert" \
--use_xformer --canonical  --anneal --use_perceptual_loss --perceptual_weight 5e2 --use_conformal_loss \
--use_transition_loss --level_of_cc 2 \
--difficulty 'hard' --schedule_rate 6.0