%% COMPLETE SETUP GUIDE FOR VIDEO OBJECT SEGMENTATION
% Follow these steps exactly to set up and run the system

%% STEP 1: CREATE FOLDER STRUCTURE
% Create a new folder on your computer called: VideoSegmentation
% Inside this folder, you will create 5 files

%% STEP 2: CREATE THE FILES
% Copy each code section below into separate .m files with the exact names shown

% ==============================================================================
% FILE 1: AdvancedVideoSegmentationAlgorithm.m
% ==============================================================================
% Copy the ENTIRE first artifact (the algorithm class) into this file
% This file should start with: classdef AdvancedVideoSegmentationAlgorithm < vision.labeler.AutomationAlgorithm

% ==============================================================================  
% FILE 2: VideoSegmentationEvaluator.m
% ==============================================================================
% Copy the ENTIRE second artifact (the evaluation class) into this file
% This file should start with: classdef VideoSegmentationEvaluator < handle

% ==============================================================================
% FILE 3: demo_video_segmentation.m 
% ==============================================================================
% Copy the ENTIRE third artifact (the demo script) into this file
% This file should start with: function demo_video_segmentation()

% ==============================================================================
% FILE 4: main_video_segmentation.m
% ==============================================================================
% Copy the code below (the updated main script with visualization):

clear; clc; close all;

fprintf('=== Video Object Segmentation System ===\n\n');

% Check if required files exist
checkRequiredFiles();

% Run the main system
runVideoSegmentationSystem();

%% Function to check required files
function checkRequiredFiles()
    fprintf('Checking required files...\n');
    
    requiredFiles = {
        'AdvancedVideoSegmentationAlgorithm.m',
        'VideoSegmentationEvaluator.m',
        'demo_video_segmentation.m'
    };
    
    allFilesExist = true;
    for i = 1:length(requiredFiles)
        if exist(requiredFiles{i}, 'file')
            fprintf('✓ Found: %s\n', requiredFiles{i});
        else
            fprintf('✗ Missing: %s\n', requiredFiles{i});
            allFilesExist = false;
        end
    end
    
    if allFilesExist
        fprintf('✓ All required files found!\n\n');
    else
        fprintf('\n❌ Please create the missing files before proceeding.\n');
        error('Missing required files. Please create them first.');
    end
end

%% Main system function
function runVideoSegmentationSystem()
    while true
        fprintf('Choose execution mode:\n');
        fprintf('1. Quick Demo (synthetic data)\n');
        fprintf('2. DAVIS 2017 Evaluation\n');
        fprintf('3. Custom Video with Visualization (RECOMMENDED)\n');
        fprintf('4. Video Labeler Integration Setup\n');
        fprintf('5. Parameter Tuning\n');
        fprintf('6. Exit\n');
        
        choice = input('Enter your choice (1-6): ');
        fprintf('\n');
        
        switch choice
            case 1
                runQuickDemo();
            case 2
                runDAVISEvaluation();
            case 3
                runCustomVideoProcessingWithVisualization();
            case 4
                setupVideoLabelerIntegration();
            case 5
                runParameterTuning();
            case 6
                fprintf('Goodbye!\n');
                break;
            otherwise
                fprintf('Invalid choice. Please try again.\n\n');
        end
        
        fprintf('\n');
    end
end

%% Mode 1: Quick Demo
function runQuickDemo()
    fprintf('=== Running Quick Demo ===\n');
    
    try
        if ~exist('demo_video_segmentation.m', 'file')
            fprintf('❌ demo_video_segmentation.m not found.\n');
            return;
        end
        
        fprintf('Starting demo with synthetic data...\n');
        demo_video_segmentation;
        fprintf('✓ Quick demo completed successfully!\n');
        
    catch ME
        fprintf('❌ Error in quick demo: %s\n', ME.message);
    end
end

%% Mode 2: DAVIS 2017 Evaluation  
function runDAVISEvaluation()
    fprintf('=== DAVIS 2017 Dataset Evaluation ===\n');
    
    davisPath = input('Enter path to DAVIS 2017 dataset (or press Enter to skip): ', 's');
    
    if isempty(davisPath)
        fprintf('Skipping DAVIS evaluation.\n');
        return;
    end
    
    if ~exist(davisPath, 'dir')
        fprintf('❌ Dataset path does not exist: %s\n', davisPath);
        return;
    end
    
    try
        fprintf('Initializing algorithm and evaluator...\n');
        algorithm = AdvancedVideoSegmentationAlgorithm();
        evaluator = VideoSegmentationEvaluator(algorithm);
        
        outputDir = './davis_evaluation_results';
        fprintf('Running evaluation. Results will be saved to: %s\n', outputDir);
        
        evaluator.runComprehensiveEvaluation(davisPath, outputDir);
        fprintf('✓ DAVIS evaluation completed!\n');
        
    catch ME
        fprintf('❌ Error in DAVIS evaluation: %s\n', ME.message);
    end
end

%% Mode 3: Custom Video Processing with Full Visualization
function runCustomVideoProcessingWithVisualization()
    fprintf('=== Custom Video Processing with Visualization ===\n');
    
    % Get video file
    [videoFile, videoPath] = uigetfile({
        '*.mp4', 'MP4 Files (*.mp4)';
        '*.avi', 'AVI Files (*.avi)';
        '*.mov', 'MOV Files (*.mov)';
        '*.mkv', 'MKV Files (*.mkv)';
        '*.*', 'All Files (*.*)'
    }, 'Select Video File for Segmentation');
    
    if isequal(videoFile, 0)
        fprintf('No video selected.\n');
        return;
    end
    
    videoFullPath = fullfile(videoPath, videoFile);
    fprintf('Selected: %s\n', videoFile);
    
    try
        % Load video
        videoReader = VideoReader(videoFullPath);
        
        fprintf('\nVideo Properties:\n');
        fprintf('  Resolution: %dx%d\n', videoReader.Width, videoReader.Height);
        fprintf('  Frame Rate: %.2f fps\n', videoReader.FrameRate);
        fprintf('  Duration: %.2f seconds\n', videoReader.Duration);
        fprintf('  Total Frames: %d\n', videoReader.NumFrames);
        
        % Limit frames for demo
        maxFrames = min(50, videoReader.NumFrames);
        if videoReader.NumFrames > 50
            fprintf('Processing first %d frames for demo.\n', maxFrames);
        end
        
        % Get initial segmentation
        firstFrame = readFrame(videoReader);
        initialMask = getInitialMaskFromUser(firstFrame);
        
        if sum(initialMask(:)) == 0
            fprintf('No initial segmentation provided. Exiting.\n');
            return;
        end
        
        % Process video with visualization
        processVideoWithVisualization(videoReader, initialMask, maxFrames);
        
    catch ME
        fprintf('Error processing video: %s\n', ME.message);
    end
end

function mask = getInitialMaskFromUser(firstFrame)
    fprintf('\nProvide initial segmentation:\n');
    fprintf('1. Draw rectangle around object\n');
    fprintf('2. Click center point (will create circle)\n');
    fprintf('3. Use center region automatically\n');
    
    choice = input('Choose method (1-3): ');
    
    [h, w, ~] = size(firstFrame);
    mask = false(h, w);
    
    figure('Name', 'Initial Segmentation', 'Position', [100, 100, 800, 600]);
    imshow(firstFrame);
    title('Create Initial Segmentation');
    
    switch choice
        case 1
            fprintf('Draw a rectangle around the object (click and drag)...\n');
            rect = getrect();
            if ~isempty(rect) && all(rect > 0)
                x1 = max(1, round(rect(1)));
                y1 = max(1, round(rect(2)));
                x2 = min(w, round(rect(1) + rect(3)));
                y2 = min(h, round(rect(2) + rect(4)));
                mask(y1:y2, x1:x2) = true;
            end
            
        case 2
            fprintf('Click on the object center...\n');
            [x, y] = ginput(1);
            if ~isempty(x)
                radius = input('Enter radius (pixels, default 40): ');
                if isempty(radius), radius = 40; end
                [X, Y] = meshgrid(1:w, 1:h);
                mask = sqrt((X - x).^2 + (Y - y).^2) <= radius;
            end
            
        otherwise
            fprintf('Using center region automatically...\n');
            centerX = w/2; centerY = h/2;
            radius = min(w, h) / 6;
            [X, Y] = meshgrid(1:w, 1:h);
            mask = sqrt((X - centerX).^2 + (Y - centerY).^2) <= radius;
    end
    
    % Show initial mask
    hold on;
    contour(mask, [0.5, 0.5], 'r-', 'LineWidth', 3);
    title('Initial Segmentation (Red outline) - Close window when satisfied');
    
    fprintf('Close the figure window when you are satisfied with the segmentation.\n');
    waitfor(gcf);
end

function processVideoWithVisualization(videoReader, initialMask, maxFrames)
    fprintf('\nProcessing video with real-time visualization...\n');
    
    % Setup visualization
    fig = figure('Name', 'Video Segmentation - Real-time', ...
                'Position', [100, 100, videoReader.Width+100, videoReader.Height+150]);
    ax = axes('Parent', fig, 'Position', [0.05, 0.15, 0.9, 0.8]);
    
    % Add frame counter
    frameText = uicontrol('Style', 'text', 'String', 'Frame: 1', ...
                         'Position', [10, 80, 100, 20], 'FontSize', 12);
    progressText = uicontrol('Style', 'text', 'String', 'Progress: 0%', ...
                            'Position', [10, 60, 150, 20], 'FontSize', 12);
    speedText = uicontrol('Style', 'text', 'String', 'Speed: 0.000 s/frame', ...
                         'Position', [10, 40, 200, 20], 'FontSize', 12);
    
    % Initialize storage
    results = struct();
    results.masks = false(videoReader.Height, videoReader.Width, maxFrames);
    results.processingTimes = zeros(maxFrames, 1);
    results.objectSizes = zeros(maxFrames, 1);
    
    % Reset video
    videoReader.CurrentTime = 0;
    
    % Process frames
    previousFrame = readFrame(videoReader);
    previousMask = initialMask;
    results.masks(:,:,1) = initialMask;
    results.objectSizes(1) = sum(initialMask(:));
    
    for frameIdx = 1:maxFrames
        tic;
        
        % Read frame
        if frameIdx == 1
            currentFrame = previousFrame;
            currentMask = initialMask;
        else
            if hasFrame(videoReader)
                currentFrame = readFrame(videoReader);
                currentMask = trackObjectInFrame(previousFrame, currentFrame, previousMask);
            else
                break;
            end
        end
        
        % Store results
        results.masks(:,:,frameIdx) = currentMask;
        results.processingTimes(frameIdx) = toc;
        results.objectSizes(frameIdx) = sum(currentMask(:));
        
        % Update visualization
        overlayImage = labeloverlay(currentFrame, currentMask, 'Colormap', [1 0 0], 'Transparency', 0.3);
        
        if isempty(get(ax, 'Children'))
            imshow(overlayImage, 'Parent', ax);
        else
            im = findobj(ax, 'Type', 'Image');
            if ~isempty(im)
                set(im(1), 'CData', overlayImage);
            end
        end
        
        title(ax, sprintf('Frame %d - Real-time Segmentation', frameIdx));
        
        % Update info displays
        set(frameText, 'String', sprintf('Frame: %d/%d', frameIdx, maxFrames));
        set(progressText, 'String', sprintf('Progress: %.1f%%', frameIdx/maxFrames*100));
        set(speedText, 'String', sprintf('Speed: %.3f s/frame', results.processingTimes(frameIdx)));
        
        drawnow;
        
        % Progress
        if mod(frameIdx, 10) == 0
            fprintf('  Frame %d/%d processed (%.1f%% complete)\n', frameIdx, maxFrames, frameIdx/maxFrames*100);
        end
        
        % Update for next iteration
        previousFrame = currentFrame;
        previousMask = currentMask;
        
        % Small pause for visualization
        pause(0.05);
    end
    
    % Show results
    showResultsSummary(results, maxFrames);
    saveSegmentationResults(videoReader, results, maxFrames);
end

function newMask = trackObjectInFrame(prevFrame, currFrame, prevMask)
    % Simple but effective object tracking
    
    if sum(prevMask(:)) == 0
        newMask = prevMask;
        return;
    end
    
    % Get object properties
    [maskY, maskX] = find(prevMask);
    centerX = round(mean(maskX));
    centerY = round(mean(maskY));
    
    % Template matching
    templateSize = 50;
    halfSize = templateSize/2;
    [h, w, ~] = size(prevFrame);
    
    % Extract template
    x1 = max(1, centerX - halfSize);
    x2 = min(w, centerX + halfSize);
    y1 = max(1, centerY - halfSize);
    y2 = min(h, centerY + halfSize);
    
    if x2 <= x1 || y2 <= y1
        newMask = prevMask;
        return;
    end
    
    prevTemplate = rgb2gray(prevFrame(y1:y2, x1:x2, :));
    
    % Search for best match
    searchRange = 25;
    bestCorr = -inf;
    bestDx = 0;
    bestDy = 0;
    
    for dx = -searchRange:5:searchRange
        for dy = -searchRange:5:searchRange
            newCenterX = centerX + dx;
            newCenterY = centerY + dy;
            
            nx1 = max(1, newCenterX - halfSize);
            nx2 = min(w, newCenterX + halfSize);
            ny1 = max(1, newCenterY - halfSize);
            ny2 = min(h, newCenterY + halfSize);
            
            if nx2-nx1 == x2-x1 && ny2-ny1 == y2-y1
                currTemplate = rgb2gray(currFrame(ny1:ny2, nx1:nx2, :));
                if size(currTemplate, 1) == size(prevTemplate, 1) && ...
                   size(currTemplate, 2) == size(prevTemplate, 2)
                    correlation = corr2(prevTemplate, currTemplate);
                    if correlation > bestCorr
                        bestCorr = correlation;
                        bestDx = dx;
                        bestDy = dy;
                    end
                end
            end
        end
    end
    
    % Apply motion to mask
    newMask = false(h, w);
    for i = 1:length(maskX)
        newX = maskX(i) + bestDx;
        newY = maskY(i) + bestDy;
        
        if newX >= 1 && newX <= w && newY >= 1 && newY <= h
            newMask(newY, newX) = true;
        end
    end
    
    % Morphological cleanup
    se = strel('disk', 2);
    newMask = imclose(newMask, se);
    newMask = imopen(newMask, se);
end

function showResultsSummary(results, numFrames)
    fprintf('\n=== Segmentation Results ===\n');
    fprintf('Frames processed: %d\n', numFrames);
    fprintf('Average processing time: %.3f seconds/frame\n', mean(results.processingTimes(1:numFrames)));
    fprintf('Total processing time: %.2f seconds\n', sum(results.processingTimes(1:numFrames)));
    
    avgSize = mean(results.objectSizes(1:numFrames));
    sizeStd = std(results.objectSizes(1:numFrames));
    fprintf('Average object size: %.1f ± %.1f pixels\n', avgSize, sizeStd);
    
    % Create performance plot
    figure('Name', 'Segmentation Performance', 'Position', [500, 300, 1000, 400]);
    
    subplot(1, 3, 1);
    plot(1:numFrames, results.processingTimes(1:numFrames), 'b-', 'LineWidth', 2);
    xlabel('Frame Number');
    ylabel('Processing Time (s)');
    title('Processing Speed');
    grid on;
    
    subplot(1, 3, 2);
    plot(1:numFrames, results.objectSizes(1:numFrames), 'r-', 'LineWidth', 2);
    xlabel('Frame Number');
    ylabel('Object Size (pixels)');
    title('Object Size Over Time');
    grid on;
    
    subplot(1, 3, 3);
    if numFrames > 1
        sizeDiff = abs(diff(results.objectSizes(1:numFrames)));
        plot(2:numFrames, sizeDiff, 'g-', 'LineWidth', 2);
        xlabel('Frame Number');
        ylabel('Size Change (pixels)');
        title('Temporal Stability');
        grid on;
    end
    
    sgtitle('Video Segmentation Analysis');
end

function saveSegmentationResults(videoReader, results, numFrames)
    [~, videoName, ~] = fileparts(videoReader.Name);
    outputDir = sprintf('segmentation_%s_%s', videoName, datestr(now, 'yyyymmdd_HHMMSS'));
    
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    
    fprintf('\nSaving results to: %s\n', outputDir);
    
    % Save masks
    for frameIdx = 1:numFrames
        maskFile = fullfile(outputDir, sprintf('mask_%03d.png', frameIdx));
        imwrite(results.masks(:,:,frameIdx), maskFile);
    end
    
    % Save data
    resultsFile = fullfile(outputDir, 'results.mat');
    save(resultsFile, 'results', 'numFrames');
    
    fprintf('✓ Results saved successfully!\n');
end

%% Mode 4: Video Labeler Integration
function setupVideoLabelerIntegration()
    fprintf('=== Video Labeler Integration Setup ===\n');
    fprintf('Creating VideoLabelerAutomation.m file...\n');
    
    automationCode = [
        'classdef VideoLabelerAutomation < vision.labeler.AutomationAlgorithm\n', ...
        '    properties(Constant)\n', ...
        '        Name = ''Advanced Video Segmentation'';\n', ...
        '        Description = ''Multi-technique video object segmentation'';\n', ...
        '        UserDirections = ''Draw initial annotation on first frame'';\n', ...
        '    end\n', ...
        '    \n', ...
        '    properties\n', ...
        '        Algorithm\n', ...
        '    end\n', ...
        '    \n', ...
        '    methods\n', ...
        '        function this = VideoLabelerAutomation()\n', ...
        '            this.Algorithm = AdvancedVideoSegmentationAlgorithm();\n', ...
        '        end\n', ...
        '        \n', ...
        '        function isValid = checkSetup(this)\n', ...
        '            isValid = this.Algorithm.checkSetup();\n', ...
        '        end\n', ...
        '        \n', ...
        '        function automatedLabels = run(this, videoReader, selectedLabelName, existingLabels, currentFrame)\n', ...
        '            automatedLabels = this.Algorithm.run(videoReader, selectedLabelName, existingLabels, currentFrame);\n', ...
        '        end\n', ...
        '    end\n', ...
        'end'
    ];
    
    fid = fopen('VideoLabelerAutomation.m', 'w');
    fprintf(fid, '%s', automationCode);
    fclose(fid);
    
    fprintf('✓ VideoLabelerAutomation.m created!\n\n');
    fprintf('To use with Video Labeler:\n');
    fprintf('1. Run: videoLabeler\n');
    fprintf('2. Load your video\n');
    fprintf('3. Create pixel labels\n');
    fprintf('4. Use Algorithm > "Advanced Video Segmentation"\n');
end

%% Mode 5: Parameter Tuning
function runParameterTuning()
    fprintf('=== Parameter Tuning ===\n');
    fprintf('This algorithm uses classical computer vision.\n');
    fprintf('No training required - ready to use!\n\n');
    fprintf('You can adjust parameters like:\n');
    fprintf('- Search radius for tracking\n');
    fprintf('- Template matching thresholds\n');
    fprintf('- Color similarity weights\n');
end

% ==============================================================================
% FILE 5: setup_instructions.m
% ==============================================================================
% This file contains the setup instructions (copy the code below):

%% SETUP INSTRUCTIONS - READ THIS FIRST!

% FOLDER STRUCTURE:
% Create a folder called "VideoSegmentation" and put these 5 files in it:
% 
% VideoSegmentation/
% ├── AdvancedVideoSegmentationAlgorithm.m    (Copy from Artifact 1)
% ├── VideoSegmentationEvaluator.m            (Copy from Artifact 2) 
% ├── demo_video_segmentation.m               (Copy from Artifact 3)
% ├── main_video_segmentation.m               (Copy from above code)
% └── setup_instructions.m                    (This file)

% HOW TO RUN:
% 1. Open MATLAB
% 2. Navigate to your VideoSegmentation folder: cd('C:\your\path\VideoSegmentation')
% 3. Run: main_video_segmentation
% 4. Choose Option 3 for custom video with visualization

% WHAT YOU NEED:
% - MATLAB with Computer Vision Toolbox
% - A video file (mp4, avi, mov, etc.)
% - Nothing else! No training data needed.

% RECOMMENDED FIRST RUN:
% 1. Run Option 1 (Quick Demo) first to test everything works
% 2. Then run Option 3 with your own video

fprintf('Setup Instructions Loaded!\n');
fprintf('Follow the comments above to set up your video segmentation system.\n\n');

fprintf('QUICK START:\n');
fprintf('1. Create folder: VideoSegmentation\n');
fprintf('2. Copy the 5 code files into it\n');
fprintf('3. Open MATLAB and navigate to the folder\n');
fprintf('4. Run: main_video_segmentation\n');
fprintf('5. Choose Option 3 for custom video processing\n\n');

fprintf('The system will:\n');
fprintf('✓ Let you select a video file\n');
fprintf('✓ Draw initial segmentation on first frame\n');
fprintf('✓ Show real-time tracking with red overlay\n');
fprintf('✓ Display performance metrics\n');
fprintf('✓ Save all results automatically\n');