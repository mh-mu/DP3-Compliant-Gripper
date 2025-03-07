import numpy as np
import cv2

class DummyEnv:
    def __init__(self, goal_position, force_coefficient=1e-2):
        self.grid_shape = (480, 640)

        assert isinstance(goal_position, np.ndarray), "Goal position must be a numpy array"
        assert len(goal_position) == 2, "Goal position must be of shape (2,)"

        self.goal_position = goal_position
        self.force_coefficient = force_coefficient
        self.done_radius = 5

        self.latest_action = np.array([0, 0])

    def render(self):
        '''
        returns rendered image in the shape of (H x W x C) normalized between 0-1(float16)
        '''
        image = np.zeros((self.grid_shape[0], self.grid_shape[1], 3), dtype=np.uint8)
        cv2.circle(image, tuple(self.goal_position), 10, (0, 255, 0), -1)
        
        # Draw a cross at the center of the image using cv2.drawMarker
        center_x, center_y = self.grid_shape[1] // 2, self.grid_shape[0] // 2
        cv2.drawMarker(image, (center_x, center_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        return image

    def get_goal_position(self):
        return self.goal_position
        
    def set_goal_position(self, goal_position):
        assert isinstance(goal_position, np.ndarray), "Goal position must be a numpy array"
        assert len(goal_position) == 2, "Goal position must be of shape (2,)"
        if not (0 <= goal_position[0] < self.grid_shape[1] and 0 <= goal_position[1] < self.grid_shape[0]):
            raise ValueError("Goal position must be within the grid boundaries (480, 640)")
        self.goal_position = goal_position
        return self.goal_position
        
    def get_force(self):
        center_x, center_y = self.grid_shape[1] // 2, self.grid_shape[0] // 2
        dx = self.goal_position[0] - center_x
        dy = self.goal_position[1] - center_y
        
        force_x = self.force_coefficient * dx
        force_y = self.force_coefficient * dy
        
        return np.array([force_x, force_y])

    def take_action(self, action):
        '''
        action is a numpy array of shape (2,) representing the change in position
        '''
        assert isinstance(action, np.ndarray), "Action must be a numpy array"
        assert action.shape == (2,), "Action must be of shape (2,)"

        self.latest_action = action        
        new_position = self.goal_position + action
        
        # Ensure the new position is within the grid boundaries
        new_position = np.array([
            max(0, min(new_position[0], self.grid_shape[1] - 1)),
            max(0, min(new_position[1], self.grid_shape[0] - 1))
        ], dtype=np.int32)
        self.goal_position = new_position

    def get_latest_action(self):
        return self.latest_action
    
    def calculate_action_towards_goal(self, initial_action_size=1):
        center_x, center_y = self.grid_shape[1] // 2, self.grid_shape[0] // 2
        direction = np.array([center_x, center_y]) - self.goal_position
        distance = np.linalg.norm(direction)
        
        if distance == 0:
            return np.array([0, 0])
        
        # Scale the action size based on the distance to the goal
        action_size = max(1, initial_action_size * (distance / np.linalg.norm([center_x, center_y])))
        direction_normalized = direction / distance
        action = direction_normalized * action_size
        
        return action.astype(np.int32)
    
    def step(self, action):
        self.take_action(action)

        center_x, center_y = self.grid_shape[1] // 2, self.grid_shape[0] // 2
        distance_to_center = np.linalg.norm(self.goal_position - np.array([center_x, center_y]))
        done = distance_to_center <= self.done_radius
        return done
    
    def reset(self):
        self.goal_position = np.array([
            np.random.randint(0, self.grid_shape[1]),
            np.random.randint(0, self.grid_shape[0])
        ])
        self.latest_action = np.array([0, 0])
        return self.goal_position

# Example usage
if __name__ == "__main__":
    goal_pos = (20, 60)
    env = DummyEnv(goal_pos)
    rendered_image = env.render()
    cv2.imshow("Rendered Image", rendered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Goal Position:", env.get_goal_position())
    env.take_action((10, 15))
    print("Goal Position after action:", env.get_goal_position())
