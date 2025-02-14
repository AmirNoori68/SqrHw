clear; tic
close all
%%
% Sphere 
dsites=[1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1;0 0 -1];
intnode=[0 0 0 ];
neval=60; 
%%
%%%%%%%%%% Evaluation points
bmin=min(dsites,[],1);  bmax=max(dsites,[],1);
xgrid=linspace(bmin(1)-.05,bmax(1)+.05,neval);
ygrid=linspace(bmin(2)-.05,bmax(2)+.05,neval);
zgrid=linspace(bmin(3)-.05,bmax(3)+.05,neval);
[xe,ye,ze]=meshgrid(xgrid,ygrid,zgrid);
epoints=[xe(:),ye(:),ze(:)]; %clear xgrid ygrid zgrid
%%
% Create a reference sphere
[x_sphere, y_sphere, z_sphere] = sphere(100);  % Sphere with radius 1 and center at (0, 0, 0)
% Create a reference circle
theta = linspace(0, 2*pi, 100);
x_circle = cos(theta);
y_circle = sin(theta);
%%
save('griddata.mat','epoints')
%%
load('pf_data1.mat')  
% Access the loaded variables
% pf = pfa;  % Access the 'pf1' variable
pf_list = pfa_list';
pf_list = pf_list(:,:);
% pf_list = pf_list(:,10);

%% 2D
% Set up the video writer
outputVideo = VideoWriter('sphere2D1.avi');
outputVideo.FrameRate = 20; % Adjust frame rate (frames per second)
open(outputVideo);

% Iterate through columns and update the plot
for col = 1:size(pf_list, 2)
    clf;  % Clear the figure

    pf = pf_list(:, col);
    % Create 3D plot
    pfit = patch(isosurface(xe, ye, ze, reshape(pf, neval, neval, neval), 0));
    isonormals(xe, ye, ze, reshape(pf, neval, neval, neval), pfit);
    hold on;    
    set(pfit, 'FaceLighting', 'gouraud', 'FaceColor', 'g', 'EdgeColor', 'none');
    daspect([1 1 1]);
    light('Position', [15 0 0], 'Style', 'infinite');
    light('Position', [-15 0 0], 'Style', 'local');
    title(['Epoch ', num2str(col)], 'FontSize', 18);
%     title('Epoch 50', 'FontSize', 18);
    % Plot the reference circle
    plot(x_circle, y_circle, 'b--', 'LineWidth', 1);
    xlabel('x');
    ylabel('y');
%     xlim([-1 1]);ylim([-1 1]);
    % Capture the frame and add it to the video
    frame = getframe(gcf);
    writeVideo(outputVideo, frame);
    
    % Pause to allow time for the animation
%     pause(0.1);
    
   axis equal
end
% Close the video writer
close(outputVideo);
%% 3D
% Set up the video writer
% outputVideo = VideoWriter('sphere3D.avi', 'Uncompressed AVI');
outputVideo = VideoWriter('sphere3D1.avi');
outputVideo.FrameRate = 30; % Adjust frame rate (frames per second)
open(outputVideo);

% Create a figure
fig = figure;

% Iterate through columns and update the plot
for col = 1:size(pf_list, 2)
    clf;  % Clear the figure
     surf(x_sphere, y_sphere, z_sphere, 'EdgeAlpha', 0.5, 'EdgeColor', 'red', 'FaceColor', 'none');
%     view([-45,-45]); 

    hold on
 
    pf = pf_list(:, col);
    
    % Create 3D plot
    pfit = patch(isosurface(xe, ye, ze, reshape(pf, neval, neval, neval), 0));
    isonormals(xe, ye, ze, reshape(pf, neval, neval, neval), pfit);
    set(pfit, 'FaceLighting', 'gouraud', 'FaceColor', 'g', 'EdgeColor', 'none');
    daspect([1 1 1]);
    light('Position', [15 0 0], 'Style', 'infinite');
    light('Position', [-15 0 0], 'Style', 'local');
    title(['Epoch ', num2str(col)], 'FontSize', 18);
%     title('Epoch 400', 'FontSize', 18);
    xlabel('x');
    ylabel('y');
    zlabel('z');
    % Plot the reference sphere
    
%     view([-45,-45]); 
    
    % Capture the frame and add it to the video
    frame = getframe(gcf);
    writeVideo(outputVideo, frame);
    
    % Pause to allow time for the animation
%     pause(0.1);
% colormap jet
set(gca,'FontSize',14);

   axis equal

end

% Close the video writer
close(outputVideo);