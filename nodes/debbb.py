from stable_baselines3 import SAC

model_path = "./sac_donkeycar_checkpoints/sac_donkeycar_250000_steps.zip"
#model_path = "./final_models/Model_try_1_normalized.zip"
SAC.load(model_path)
print("success")
