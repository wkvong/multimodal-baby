SCRIPTS_DIR=/scratch/nk3351/projects/multimodal-baby/slurm
EMAIL=najoung.kim@nyu.edu
LOG_DIR=/scratch/nk3351/projects/multimodal-baby/logs

python runner.py \
   --scripts ${SCRIPTS_DIR} \
   --log ${LOG_DIR} \
   --mail-type END,FAIL \
   --mail-user ${EMAIL} \
   --python python \
   --time 1-0:00 \
   --basename joint \
   --config runner_config/saycam_joint.py

python runner.py \
   --scripts ${SCRIPTS_DIR} \
   --log ${LOG_DIR} \
   --mail-type END,FAIL \
   --mail-user ${EMAIL} \
   --python python \
   --time 1-0:00 \
   --basename lm \
   --config runner_config/saycam_lm.py

python runner.py \
   --scripts ${SCRIPTS_DIR} \
   --log ${LOG_DIR} \
   --mail-type END,FAIL \
   --mail-user ${EMAIL} \
   --python python \
   --time 1-0:00 \
   --basename multimodal \
   --config runner_config/saycam_multimodal.py



