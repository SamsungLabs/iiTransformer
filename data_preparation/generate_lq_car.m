SRC_PATH = 'DB/Classic5/GT';

filepaths = dir(fullfile(SRC_PATH,'*.*'));
for i = 3:length(filepaths)
    img_name = filepaths(i).name;

    % Read image
    img = im2double(imread([SRC_PATH, img_name]));

    % convert to uint8
    if max(img(:)) < 1
        img = uint8(round(255*img));
        img(img < 0) = 0;
        img(img > 255) = 255;
        img = uint8(img);
    end

    %% convert to grayscale
    if ndims(img) == 3
        img_gray = rgb2gray(img);
        DST_PATH = 'DB/Classic5/LQ/gray_matlab';
        imwrite(img_gray, fullfile(DST_PATH, img_name), 'jpeg', 'Quality', 100);
    else
        img_gray = img;
    end
     
     %% jpeg
     img_name_noExt = split(img_name, '.');
     img_name_noExt = img_name_noExt{1};
     for JPEG_Quality = 10:10:40
         DST_PATH = sprintf('DB/Classic5/LQ/jpeg_q%02d_matlab', JPEG_Quality);
         imwrite(img, fullfile(DST_PATH, [img_name_noExt, '.jpg']), 'jpeg', 'Quality', JPEG_Quality);
 
         % read compressed image and convert to grayscale
         img_cmpr = imread(fullfile(DST_PATH, [img_name_noExt '.jpg']));
         if ndims(img_cmpr) == 3
             img_cmpr = rgb2gray(img_cmpr);
         end
         DST_PATH = sprintf('DB/Classic5/LQ/jpeg_q%02d_gray_matlab', JPEG_Quality);
         imwrite(img_cmpr, fullfile(DST_PATH, [img_name_noExt, '.png']));
     end

end
