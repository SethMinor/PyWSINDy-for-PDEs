% Chebfun script for integrating SH23 model
% (See chebfun.org/examples/pde/SwiftHohenberg.html)
clearvars, clc;

% Specify simulation details
Ngrid = 200;
dt = 0.2;
domain = [0, 30*pi, 0, 30*pi];
t = 0:dt:20;
S = spinop2(domain, t);

% Define parameters
r = 0.4;
d = 1;
S.lin =@(u) -2*lap(u) - biharm(u);
S.nonlin =@(u) (r-1)*u + d*u.^2 - u.^3;

% Initial condition (smooth random function)
lambda = 2*pi; % minimal wavelength
u0 = randnfun2(lambda, 'big', 'trig', domain);
S.init = u0;

% Run the simulation
u = spin2(S, Ngrid, dt);

% Save the simulation data
u_storage = zeros(Ngrid * Ngrid, length(t));
[Xgrid, Ygrid] = meshgrid(linspace(domain(1), domain(2), Ngrid), ...
                          linspace(domain(3), domain(4), Ngrid));

% Loop through each time step and store each u(t) as a column
for ti = 1:length(t)
    % Evaluate u at the grid points for the current time step
    u_vals = feval(u{ti}, Xgrid, Ygrid);
    
    % Store the u values in the matrix
    u_storage(:, ti) = u_vals(:);
end

% Write the csv file
writematrix(u_storage,'SH23');

%% Save a video

v = VideoWriter('sh23_simulation.avi');
v.Quality = 85;
open(v)
figure (2);
for k = 1:length(u)
    plot(u{k});
    view(0,90)
    daspect([1 1 1])
    axis off
    frame = getframe(gcf);
    writeVideo(v, frame);
end
close(v)
