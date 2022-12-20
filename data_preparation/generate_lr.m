SRC_PATH = 'DB/Urban100/GT/';
crop = true;
if crop
    DST_PATH = 'DB/Urban100/GTmod12';
    mkdir(DST_PATH)
end

for factor = 2:4
    scale = 1/factor;
    
    DST_PATH = sprintf('DB/Urban100/LQ/x%d_matlab_mod12', factor);
    mkdir(DST_PATH)

    filepaths = dir(fullfile(SRC_PATH,'*.*'));
    for i = 3:length(filepaths)
        [~, img_name, ext] = fileparts(filepaths(i).name);

        % Read image
        img = im2double(imread([SRC_PATH, strcat(img_name, ext)]));
        
        % Crop image to modulo 12 (for compatibility with x3 and x4)
        if crop
            [height, width, ch] = size(img);
            height_new = height - mod(height, 12);
            width_new = width - mod(width, 12);
            img = img(1:height_new, 1:width_new, :);
            imwrite(img, fullfile(DST_PATH, strcat(img_name, ext)))
        end

        % Resize image using default settings (i.e., bicubic interp)
        img_lr = imresize(img, scale);

        %% Save image
        if strcmpi(ext, '.jpg') || strcmpi(ext, '.jpeg')
            imwrite(img_lr, fullfile(DST_PATH, strcat(img_name, ext)), 'Quality', 100);
        else
            imwrite(img_lr, fullfile(DST_PATH, strcat(img_name, ext)));
        end

    end
end