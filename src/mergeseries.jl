using DataFrames

data = readcsv("data.csv")

n_assets = Int(size(data,2)/2)
n_lines = size(data,1)

date = Array(String,n_lines)
data_out = Array(Float64,n_lines,n_assets)
col_assets = collect(2:2:10)

n_lines_out = 1
curr_out =  Array(Float64,n_assets)
for line = 1:size(data,1)
  curr_date = data[line,1]
  found_all = true
  for i = 1:n_assets
    id = findfirst(data[:,(col_assets-1)[i]],curr_date)
    if id != 0
      curr_out[i] = data[id,col_assets[i]]
    else
      found_all = false
      break
    end
  end
  if found_all
    data_out[n_lines_out,:] = curr_out
    date[n_lines_out] = curr_date
    n_lines_out += 1
  end
end


out_all = convert(DataFrame, data_out)
out_all[:dates] = date

out_all = out_all[1:n_lines_out-1,:]
