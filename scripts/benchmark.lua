local body_size = 224 * 224 * 3 * 4
local body = string.rep("\0", body_size)

wrk.method = "POST"
wrk.body = body
wrk.headers["Content-Type"] = "application/octet-stream"

request = function()
   return wrk.format("POST", "/predict", wrk.headers, wrk.body)
end
