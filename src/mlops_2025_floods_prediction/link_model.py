from pathlib import Path
import wandb

run = wandb.init(project="collection-linking-quickstart")

artifact_filepath = Path("./my_Select Model_artifact.txt")
artifact_filepath.write_text("simulated Select Model file")
  
logged_artifact = run.log_artifact(
  artifact_filepath,
  "artifact-name",
  type="Select Model"
)
run.link_artifact(   
  artifact=logged_artifact,  
  target_path="wandb-registry-floods_prediction/Staging Models"
)
run.finish()