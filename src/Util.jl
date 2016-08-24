module Util

export memuse

function memuse()
  pid = getpid()
  return string(round(Int,parse(Int,readall(`ps -p $pid -o rss=`))/1024),"M")
end

end # Util
