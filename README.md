# 2D-Rectangle-Packing
The 2D Rectangle Packing Problem involves efficiently placing a certain number of rectangular objects into a container with a fixed width (W) and height (H). The goal here is to maximize the usage of the container's area as much as possible. This project is a partial implementation of [1]

## Pseudo-code
    "W, H: Width & Height of the container rectangle.
    rectangles: Rectangles to be placed.
    L: List of all possible corner-occupying placements with unplaced rectangles.
    configuration: The state of container with already placed rectangles."

    Function A0(W, H, rectangles):
        Initialize configuration with 0 placed rectangles
        Initialize L with configuration
        While(L is not empty)
            for each CCOA in L:
                Calculate CCOA degree
            Select CCOA with highest degree
            Modify configuration by executing best CCOA
            Modify L with configuration
        return configuration

## References
[1] Chen, Mao, and Wenqi Huang. "A two-level search algorithm for 2D rectangular packing problem." Computers & Industrial Engineering 53.1 (2007): 123-136.
