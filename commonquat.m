function cq= commonquat(vicon_data)
    cq = zeros(length(vicon_data(:,1)), 4);
    cq(1,1:4) = [1 0 0 0];

    for j=2:length(vicon_data(:,1))
        pre = [vicon_data(j-1,1:3)' vicon_data(j-1,4:6)' vicon_data(j-1,7:9)' vicon_data(j-1,10:12)'];
        now = [vicon_data(j,1:3)' vicon_data(j,4:6)' vicon_data(j,7:9)' vicon_data(j,10:12)'];
        [regParams,Bfit,ErrorStats]=absor(pre, now);
        cq(j,1:4) = regParams.q;
    end
end