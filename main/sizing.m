
file = imread('C:\Users\david\workspace\cameraDamageSizing\main\4m\test1.jpg');

% Display one of the calibration images
magnification = 25;

imOrig = file;
figure; imshow(imOrig, 'InitialMagnification', magnification);
title('Input Image');
squareSize = 50;


[im, newOrigin] = undistortImage(imOrig, cameraParams, 'OutputView', 'full');

imagePoints1 = [2116, 1097;
                2128, 1100;
                2125, 1111;
                2114, 1109];
            
norm(imagePoints1(2, :) - imagePoints1(1, :))
norm(imagePoints1(3, :) - imagePoints1(2, :))

% Compute the diameter of the coin in millimeters.
dX = norm(imagePoints1(2, :) - imagePoints1(1, :)) / (cameraParams.IntrinsicMatrix(1, 1)) * 4000;
dY = norm(imagePoints1(3, :) - imagePoints1(2, :)) / (cameraParams.IntrinsicMatrix(2, 2)) * 4000;
diameterInMillimeters = hypot(dX, dY);
area = dX * dY;
figure; 
imshow(im, 'InitialMagnification', magnification);
rectangle('Position', [1783, 1100, 53, 53], 'EdgeColor', 'Y', 'LineWidth' ,1);
% imwrite(im, 'undistortedImageTest.jpg', 'jpg');
title('Detected box');
fprintf('Measured diagonal = %0.2f mm\n', diameterInMillimeters);
fprintf('Measured area = %0.2f mm\n', area);
fprintf('Measured side = %0.2f mm\n', dX);
fprintf('Measured side = %0.2f mm\n', dY);



