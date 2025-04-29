import clip
import torch
from PIL import Image
import cv2
import numpy as np

class CLIPReward:
    def __init__(self, goal_text="a robot arm grasping a red block"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEBUG] CLIP device: {self.device}")  # <-- Check device

        # Load CLIP model with error handling
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        except Exception as e:
            print(f"[ERROR] Failed to load CLIP model: {e}")
            raise

        self.goal_text = goal_text
        
        try:
            self.goal_features = self._encode_text(goal_text)
            print(f"[DEBUG] Goal features shape: {self.goal_features.shape}")  # Should be [1, 512]
            print(f"[DEBUG] Goal features norm: {torch.norm(self.goal_features).item()}")  # Check norm
        except Exception as e:
            print(f"[ERROR] Failed to encode goal text: {e}")
            raise
        
        self.step_count = 0 
        self.init_features = None  # Make this explicit for safety
        self.init_image = None
       

    def _encode_image(self, image):
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
        elif not image.mode == "RGB":
            image = image.convert("RGB")

        preprocessed = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(preprocessed)
            features /= features.norm(dim=-1, keepdim=True) #Normalize
        return features

    def _encode_text(self, text):
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True) #Normalize
        return features

    def compute_reward(self, image, gripper_pos, cube_pos, success=False):

        gripper_cube_dist = np.linalg.norm(np.array(gripper_pos) - np.array(cube_pos))

        # Encode initial state once
        if self.init_features is None:
            print("[CLIP] Setting initial features")
            self.init_image = image.copy()
            self.init_features = self._encode_image(self.init_image)
            return {
                'total_reward': 0.0,
                'goal_sim': 0.0,
                'init_sim': 0.0,
                'geometric_reward': 0.0
            }

        #print("[CLIP] Using existing init features")

        current_features = self._encode_image(image)

        goal_features = self.goal_features / self.goal_features.norm(dim=-1, keepdim=True)  # Explicit norm
        init_features = self.init_features / self.init_features.norm(dim=-1, keepdim=True)  # Explicit norm
        

        try:
            goal_sim = torch.cosine_similarity(current_features, goal_features, dim=1).item()
            init_sim = torch.cosine_similarity(current_features, init_features, dim=1).item()
            print(f"[CLIP] Goal sim: {goal_sim:.4f}, Init sim: {init_sim:.4f}")

        except Exception as e:
            print("[CLIP ERROR] Cosine similarity failed:", e)
            return {
                'total_reward': 0.0,
                'goal_sim': 0.0,
                'init_sim': 0.0,
                'geometric_reward': 0.0
            }

        # Calculate geometric distance reward
        #clip_reward = goal_sim
        geometric_reward = -np.linalg.norm(gripper_pos - cube_pos)

        #total_reward = 0.2 * clip_reward + 0.8 * geometric_reward  # Prioritize distance

        #Only CLIP REWARD
        #total_reward = (goal_sim - init_sim)
        total_reward = goal_sim * 10 #to promote exploring SAC_102
        
        #print(f"[DEBUG] Goal features norm: {torch.norm(self.goal_features).item():.3f}")
        #print(f"[DEBUG] Init features norm: {torch.norm(self.init_features).item():.3f}")

        if gripper_cube_dist < 0.03:  # 3cm threshold
            total_reward += 10.0 
        
        if success is True:  # When cube is grasped
            total_reward += 100.0
    
        #print("[CLIP] Goal sim:", goal_sim, "Init sim:", init_sim,
        #      "geometric reward:", geometric_reward, "total reward:", total_reward)

        # Save every 200 steps
        if self.step_count % 200 == 0:
            cv2.imwrite(f"debug/obs_{self.step_count}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.step_count += 1

        return {
            'total_reward': total_reward,
            'goal_sim': goal_sim,
            'init_sim': init_sim,
            'geometric_reward': geometric_reward
        }
