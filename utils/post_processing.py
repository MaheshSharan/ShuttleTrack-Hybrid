import numpy as np
import torch
from scipy import signal
from filterpy.kalman import KalmanFilter
from scipy.interpolate import interp1d, splrep, splev
import cv2

class TrajectoryProcessor:
    """Post-processing toolkit for shuttlecock trajectory smoothing and inpainting.
    
    Implements Section 6 of the improvement roadmap, providing trajectory 
    smoothing, visibility smoothing, and trajectory inpainting tools.
    """
    
    def __init__(self, window_size=5, mode='kalman', inpaint_method='spline', 
                 vis_threshold=0.5, vis_smooth_window=3):
        """Initialize trajectory post-processor.
        
        Args:
            window_size: Window size for smoothing filters
            mode: Smoothing method ('kalman', 'savgol', 'moving_avg')
            inpaint_method: Method for trajectory inpainting ('linear', 'spline', 'pchip')
            vis_threshold: Threshold for binary visibility prediction
            vis_smooth_window: Window size for visibility smoothing
        """
        self.window_size = window_size
        self.mode = mode
        self.inpaint_method = inpaint_method
        self.vis_threshold = vis_threshold
        self.vis_smooth_window = vis_smooth_window
        
        # Initialize Kalman filter if needed
        if mode == 'kalman':
            self.kf = None
            self._init_kalman()
    
    def _init_kalman(self):
        """Initialize Kalman filter with position and velocity state."""
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
        
        # State transition matrix (position + velocity model)
        kf.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (we only measure position)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrices - tuned for shuttlecock tracking
        kf.P *= 10  # Initial state uncertainty
        kf.R = np.eye(2) * 0.01  # Measurement uncertainty 
        
        # Process uncertainty - higher for velocity to handle quick direction changes
        kf.Q = np.diag([0.01, 0.01, 0.1, 0.1])
        
        self.kf = kf
    
    def _reset_kalman(self):
        """Reset Kalman filter state."""
        self._init_kalman()
    
    def smooth_trajectory_kalman(self, trajectory, visibility=None):
        """Smooth trajectory using Kalman filter.
        
        Args:
            trajectory: Array of shape (T, 2) with [x, y] coordinates
            visibility: Optional array of shape (T,) with visibility flags
            
        Returns:
            Smoothed trajectory of shape (T, 2)
        """
        # Reset filter state
        self._reset_kalman()
        
        T = trajectory.shape[0]
        smoothed = np.zeros_like(trajectory)
        
        # Initialize with first visible point
        first_idx = 0
        if visibility is not None:
            visible_indices = np.where(visibility > 0)[0]
            if len(visible_indices) > 0:
                first_idx = visible_indices[0]
        
        # Initialize state with first point
        self.kf.x = np.array([trajectory[first_idx, 0], trajectory[first_idx, 1], 0, 0])
        smoothed[first_idx] = trajectory[first_idx]
        
        # Forward pass
        for i in range(first_idx + 1, T):
            # Predict next state
            self.kf.predict()
            
            # Update with measurement if visible
            if visibility is None or visibility[i] > 0:
                measurement = trajectory[i]
                self.kf.update(measurement)
                
            # Get current state estimate for position
            smoothed[i] = self.kf.x[:2]
        
        return smoothed
    
    def smooth_trajectory_savgol(self, trajectory, visibility=None):
        """Smooth trajectory using Savitzky-Golay filter.
        
        Args:
            trajectory: Array of shape (T, 2) with [x, y] coordinates
            visibility: Optional array of shape (T,) with visibility flags
            
        Returns:
            Smoothed trajectory of shape (T, 2)
        """
        T = trajectory.shape[0]
        smoothed = np.copy(trajectory)
        
        # Only smooth if we have enough points
        if T >= self.window_size:
            # Set polyorder to min(window_size-1, 3) to avoid overfitting
            polyorder = min(self.window_size - 1, 3)
            
            # If we have visibility information
            if visibility is not None:
                # Only smooth visible points
                visible_indices = np.where(visibility > 0)[0]
                if len(visible_indices) >= self.window_size:
                    # Extract visible trajectory
                    visible_traj = trajectory[visible_indices]
                    
                    # Apply filter separately to x and y
                    smoothed_x = signal.savgol_filter(visible_traj[:, 0], self.window_size, polyorder)
                    smoothed_y = signal.savgol_filter(visible_traj[:, 1], self.window_size, polyorder)
                    
                    # Update visible points
                    smoothed[visible_indices, 0] = smoothed_x
                    smoothed[visible_indices, 1] = smoothed_y
            else:
                # Smooth entire trajectory
                smoothed[:, 0] = signal.savgol_filter(trajectory[:, 0], self.window_size, polyorder)
                smoothed[:, 1] = signal.savgol_filter(trajectory[:, 1], self.window_size, polyorder)
        
        return smoothed
    
    def smooth_trajectory_moving_avg(self, trajectory, visibility=None):
        """Smooth trajectory using moving average filter.
        
        Args:
            trajectory: Array of shape (T, 2) with [x, y] coordinates
            visibility: Optional array of shape (T,) with visibility flags
            
        Returns:
            Smoothed trajectory of shape (T, 2)
        """
        T = trajectory.shape[0]
        smoothed = np.copy(trajectory)
        
        # Define convolution kernel for moving average
        kernel = np.ones(self.window_size) / self.window_size
        
        # If we have visibility information
        if visibility is not None:
            # Find continuous visible segments
            visibility_binary = visibility > 0
            
            # Process each continuous segment separately
            segment_start = None
            for i in range(T):
                # Segment start
                if visibility_binary[i] and (segment_start is None):
                    segment_start = i
                
                # Segment end or last frame
                if (not visibility_binary[i] or i == T-1) and segment_start is not None:
                    segment_end = i if not visibility_binary[i] else i+1
                    
                    # Only smooth if segment is long enough
                    if segment_end - segment_start >= self.window_size:
                        for j in range(2):  # Process x and y separately
                            segment_data = trajectory[segment_start:segment_end, j]
                            # Apply convolution
                            smoothed[segment_start:segment_end, j] = np.convolve(
                                segment_data, kernel, mode='same')
                            
                            # Fix boundary effects
                            half_window = self.window_size // 2
                            smoothed[segment_start:segment_start+half_window, j] = trajectory[segment_start:segment_start+half_window, j]
                            smoothed[segment_end-half_window:segment_end, j] = trajectory[segment_end-half_window:segment_end, j]
                    
                    segment_start = None
        else:
            # Smooth entire trajectory
            for j in range(2):  # Process x and y separately
                smoothed[:, j] = np.convolve(trajectory[:, j], kernel, mode='same')
                
                # Fix boundary effects
                half_window = self.window_size // 2
                smoothed[:half_window, j] = trajectory[:half_window, j]
                smoothed[-half_window:, j] = trajectory[-half_window:, j]
        
        return smoothed
    
    def smooth_visibility(self, visibility):
        """Smooth visibility predictions with temporal filter.
        
        Args:
            visibility: Array of shape (T,) with predicted visibility scores
            
        Returns:
            Smoothed visibility scores
        """
        # Convert to numpy if needed
        if isinstance(visibility, torch.Tensor):
            visibility = visibility.detach().cpu().numpy()
            
        T = len(visibility)
        
        # Apply median filter to remove outliers
        if T >= self.vis_smooth_window:
            smoothed = signal.medfilt(visibility, self.vis_smooth_window)
        else:
            smoothed = np.copy(visibility)
            
        return smoothed
    
    def inpaint_trajectory(self, trajectory, visibility):
        """Fill in missing trajectory points during occlusions.
        
        Args:
            trajectory: Array of shape (T, 2) with [x, y] coordinates
            visibility: Array of shape (T,) with visibility flags
            
        Returns:
            Inpainted trajectory of shape (T, 2)
        """
        T = trajectory.shape[0]
        inpainted = np.copy(trajectory)
        
        # Create binary visibility mask
        visible = visibility > self.vis_threshold
        
        # If all visible or all invisible, nothing to inpaint
        if np.all(visible) or not np.any(visible):
            return inpainted
        
        # Find visible indices
        visible_indices = np.where(visible)[0]
        
        # Extract visible trajectory points
        visible_trajectory = trajectory[visible_indices]
        
        # Interpolate over the full range
        if len(visible_indices) >= 2:  # Need at least two points to interpolate
            # Create interpolator based on selected method
            if self.inpaint_method == 'linear':
                # Linear interpolation
                interp_x = interp1d(visible_indices, visible_trajectory[:, 0], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(visible_indices, visible_trajectory[:, 1], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            
            elif self.inpaint_method == 'spline':
                # Cubic spline interpolation
                interp_x = interp1d(visible_indices, visible_trajectory[:, 0], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(visible_indices, visible_trajectory[:, 1], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            elif self.inpaint_method == 'pchip':
                # PCHIP interpolation (Piecewise Cubic Hermite)
                interp_x = interp1d(visible_indices, visible_trajectory[:, 0], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
                interp_y = interp1d(visible_indices, visible_trajectory[:, 1], 
                                   kind='cubic', bounds_error=False, fill_value='extrapolate')
            
            # Generate all indices
            all_indices = np.arange(T)
            
            # Find invisible indices
            invisible_indices = np.where(~visible)[0]
            
            # Only interpolate between the first and last visible points
            interp_range = np.logical_and(
                invisible_indices >= visible_indices[0],
                invisible_indices <= visible_indices[-1]
            )
            interp_indices = invisible_indices[interp_range]
            
            # Fill in missing points
            if len(interp_indices) > 0:
                inpainted[interp_indices, 0] = interp_x(interp_indices)
                inpainted[interp_indices, 1] = interp_y(interp_indices)
        
        return inpainted
    
    def process_predictions(self, coordinates, visibility):
        """Apply complete post-processing pipeline to predictions.
        
        Args:
            coordinates: Array of shape (B, T, 2) with [x, y] coordinates
            visibility: Array of shape (B, T) with visibility scores
            
        Returns:
            Tuple of (processed_coordinates, processed_visibility)
        """
        # Convert to numpy if needed
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        if isinstance(visibility, torch.Tensor):
            visibility = visibility.detach().cpu().numpy()
            
        B, T = visibility.shape
        processed_coords = np.zeros_like(coordinates)
        processed_vis = np.zeros_like(visibility)
        
        # Process each sequence in the batch
        for b in range(B):
            # 1. Smooth visibility
            processed_vis[b] = self.smooth_visibility(visibility[b])
            
            # 2. Apply selected trajectory smoothing method
            if self.mode == 'kalman':
                smoothed = self.smooth_trajectory_kalman(coordinates[b], processed_vis[b])
            elif self.mode == 'savgol':
                smoothed = self.smooth_trajectory_savgol(coordinates[b], processed_vis[b])
            else:  # default to moving_avg
                smoothed = self.smooth_trajectory_moving_avg(coordinates[b], processed_vis[b])
                
            # 3. Apply trajectory inpainting
            processed_coords[b] = self.inpaint_trajectory(smoothed, processed_vis[b])
        
        return processed_coords, processed_vis

# Factory function to create processor based on config
def create_trajectory_processor(config):
    """Create a TrajectoryProcessor instance from config.
    
    Args:
        config: Dictionary with post-processing configuration
        
    Returns:
        TrajectoryProcessor instance
    """
    # Extract parameters from config with defaults
    post_config = config.get('post_processing', {})
    
    window_size = post_config.get('smooth_window_size', 5)
    mode = post_config.get('smooth_mode', 'kalman')
    inpaint_method = post_config.get('inpaint_method', 'spline')
    vis_threshold = post_config.get('visibility_threshold', 0.5)
    vis_smooth_window = post_config.get('visibility_smooth_window', 3)
    
    # Create and return processor
    processor = TrajectoryProcessor(
        window_size=window_size,
        mode=mode,
        inpaint_method=inpaint_method,
        vis_threshold=vis_threshold,
        vis_smooth_window=vis_smooth_window
    )
    
    return processor 