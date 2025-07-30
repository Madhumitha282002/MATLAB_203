% Set root DAVIS folder
davisRoot = fullfile(pwd, 'DAVIS');

% Define the paths
imageDir = fullfile(davisRoot, 'JPEGImages', '480p');
maskDir  = fullfile(davisRoot, 'Annotations', '480p');

% List available sequences
sequences = dir(imageDir);
sequences = sequences([sequences.isdir] & ~startsWith({sequences.name}, '.'));
disp('Available video sequences:');
disp({sequences.name}');

% Choose a sequence (example: 'bear')
sequenceName = 'bear';  

imageFiles = dir(fullfile(imageDir, sequenceName, '*.jpg'));
maskFiles  = dir(fullfile(maskDir, sequenceName, '*.png'));

% Sort files to ensure alignment
[~, idx] = sort({imageFiles.name});
imageFiles = imageFiles(idx);
[~, idx] = sort({maskFiles.name});
maskFiles = maskFiles(idx);

% Display image and mask side-by-side
for i = 1:min(length(imageFiles), length(maskFiles))
    img = imread(fullfile(imageFiles(i).folder, imageFiles(i).name));
    mask = imread(fullfile(maskFiles(i).folder, maskFiles(i).name));

    subplot(1, 2, 1); imshow(img); title(['Image: ' imageFiles(i).name]);
    subplot(1, 2, 2); imshow(mask, []); title(['Mask: ' maskFiles(i).name]);
    pause(0.1);  % adjust for animation speed
end
