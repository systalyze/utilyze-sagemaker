#include "sampler_internal.hpp"

namespace utlz_sampler_internal {

const char* GetEnv(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

bool DebugLogsEnabled() {
    static bool enabled = (GetEnv("UTLZ_SAMPLER_DEBUG") != nullptr);
    return enabled;
}

void SetError(State* s, const std::string& msg) {
    if (s) {
        std::lock_guard<std::mutex> lock(s->errorMutex);
        s->lastError = msg;
    }

    if (!DebugLogsEnabled()) return;
    std::fprintf(stderr, "[utlz:%d] %s\n", s ? s->cfg.deviceIndex : -1, msg.c_str());
}

std::vector<std::string> SplitCsv(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t b = 0, e = tok.size();
        while (b < e && std::isspace((unsigned char)tok[b])) b++;
        while (e > b && std::isspace((unsigned char)tok[e - 1])) e--;
        if (e > b) out.push_back(tok.substr(b, e - b));
    }
    return out;
}

int ReadEnvInt(const char* name, int fallback, int minVal, int maxVal) {
    const char* v = GetEnv(name);
    if (!v || !*v) return fallback;

    char* end = nullptr;
    long long x = std::strtoll(v, &end, 10);
    if (end == v) return fallback;
    if (x < minVal) x = minVal;
    if (x > maxVal) x = maxVal;
    return (int)x;
}

bool EqualsIgnoreCase(const char* a, const char* b) {
    if (!a || !b) return false;
    while (*a && *b) {
        if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

} // namespace utlz_sampler_internal
