# Game Events and Status Recognition - Super Farmer

## Overview
This project is a system for recognizing game states and events from video clips of the board game *Super Farmer*. The system detects and tracks key in-game events such as dice rolls, animal trading, predator attacks, acquiring protective dogs, and winning the game.

The outputs include real-time gameplay analysis and event detection, providing an interactive and automated experience.

## Dataset
The dataset consists of 9 video clips categorized by difficulty:
- **Easy**: Stable lighting and minimal obstructions.
- **Medium**: Variable lighting and increased shadows.
- **Hard**: Camera shake, board movement, and angled views.

Each video includes at least one of the five game events, with all events covered across datasets. 

[Dataset and Outputs](https://drive.google.com/drive/folders/1MIxwzXAEbxG8m_RI3XE0KX67lqT8TK-d?usp=sharing)

## Key Features
1. **Board Detection**: Identifies the game board under various conditions.
2. **Animal Token Detection**: Recognizes animal types using template matching and tracks game state.
3. **Dice Tracking**: Detects dice rolls and identifies results based on color and shape.
4. **Event Recognition**:
   - **Trading**: Detects tokens leaving the board.
   - **Getting a dog**: Detects a new dog appearance.
   - **Predator Attack**: Tracks changes in animal and dog states.
   - **Winning**: Recognizes when all animal slots are filled.
5. **Robustness**: Handles shadows, reflections, and movement to varying extents.

## Results
- **Easy Dataset**: High accuracy, with minor false positives for animals.
- **Medium Dataset**: Reduced accuracy due to lighting changes and shadows.
- **Hard Dataset**: Significant challenges with camera movement and angled views, leading to false detections.

## Limitations
- Difficulty in updating board positions during gameplay.
- Sensitivity to lighting and angles.
- Challenges with token and dice detection in adverse conditions.

## Techniques Used
- OpenCV for image processing (Canny edge detection, HoughCircles).
- Template matching for animal tokens.
- Region-based tracking for dogs and dice.
- Event detection based on game mechanics.

## Conclusion
The system performs well under easy conditions but struggles with harder datasets due to lighting and camera movement. Future improvements could focus on robust board position updates and angle distortion handling.

## License
This project is licensed under the MIT License.
