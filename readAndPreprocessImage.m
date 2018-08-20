    function Iout = readAndPreprocessImage(filename)

        I = imread(filename);
        if ismatrix(I)
            I = cat(3,I,I,I);
        end

        % Resize the image as required for the CNN.
        Iout = imresize(I, [227 227]);
    end
