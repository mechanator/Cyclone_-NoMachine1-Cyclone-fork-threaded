#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <omp.h>
#include <random>
#include <immintrin.h>
#include <array>
#include <mutex>

// Adding program modules
#include "sha256_avx2.h"
#include "ripemd160_avx2.h"
#include "SECP256K1.h"
#include "Point.h"
#include "Int.h"
#include "IntGroup.h"

#define BISIZE 256
#if BISIZE == 256
#define NB64BLOCK 5
#define NB32BLOCK 10
#else
#error Unsupported size
#endif

// Constants
static constexpr int POINTS_BATCH_SIZE = 256;
static constexpr int HASH_BATCH_SIZE = 8;
int g_prefixLength = 6; // Default prefix length

// Status output and progress saving frequency
static constexpr double statusIntervalSec = 5.0;
static constexpr double saveProgressIntervalSec = 300.0;

static int g_progressSaveCount = 0;
static std::vector<std::string> g_threadPrivateKeys;

// Mutex for thread-safe printing
std::mutex coutMutex;

//------------------------------------------------------------------------------
void saveProgressToFile(const std::string &progressStr) {
    std::ofstream ofs("progress.txt", std::ios::app);
    if (ofs) {
        ofs << progressStr << "\n";
    } else {
        std::cerr << "Cannot open progress.txt for writing\n";
    }
}

//------------------------------------------------------------------------------
// Converts a HEX string into a large number (a vector of 64-bit words, little-endian).
std::vector<uint64_t> hexToBigNum(const std::string &hex) {
    std::vector<uint64_t> bigNum;
    const size_t len = hex.size();
    bigNum.reserve((len + 15) / 16);
    for (size_t i = 0; i < len; i += 16) {
        size_t start = (len >= 16 + i) ? len - 16 - i : 0;
        size_t partLen = (len >= 16 + i) ? 16 : (len - i);
        uint64_t value = std::stoull(hex.substr(start, partLen), nullptr, 16);
        bigNum.push_back(value);
    }
    return bigNum;
}

// Reverse conversion to a HEX string (with correct leading zeros within blocks).
std::string bigNumToHex(const std::vector<uint64_t> &num) {
    std::ostringstream oss;
    for (auto it = num.rbegin(); it != num.rend(); ++it) {
        if (it != num.rbegin())
            oss << std::setw(16) << std::setfill('0');
        oss << std::hex << *it;
    }
    return oss.str();
}

std::vector<uint64_t> singleElementVector(uint64_t val) {
    return {val};
}

std::vector<uint64_t> bigNumAdd(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b) {
    std::vector<uint64_t> sum;
    sum.reserve(std::max(a.size(), b.size()) + 1);
    uint64_t carry = 0;
    for (size_t i = 0, sz = std::max(a.size(), b.size()); i < sz; ++i) {
        uint64_t x = (i < a.size()) ? a[i] : 0ULL;
        uint64_t y = (i < b.size()) ? b[i] : 0ULL;
        __uint128_t s = (__uint128_t)x + (__uint128_t)y + carry;
        carry = (uint64_t)(s >> 64);
        sum.push_back((uint64_t)s);
    }
    if (carry) sum.push_back(carry);
    return sum;
}

std::vector<uint64_t> bigNumSubtract(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b) {
    std::vector<uint64_t> diff = a;
    uint64_t borrow = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        uint64_t subtrahend = b[i];
        if (diff[i] < subtrahend + borrow) {
            diff[i] = diff[i] + (~0ULL) - subtrahend - borrow + 1ULL; // eqv diff[i] = diff[i] - subtrahend - borrow
            borrow = 1ULL;
        } else {
            diff[i] -= (subtrahend + borrow);
            borrow = 0ULL;
        }
    }

    for (size_t i = b.size(); i < diff.size() && borrow; ++i) {
        if (diff[i] == 0ULL) {
            diff[i] = ~0ULL;
        } else {
            diff[i] -= 1ULL;
            borrow = 0ULL;
        }
    }
    // delete leading zeros
    while (!diff.empty() && diff.back() == 0ULL)
        diff.pop_back();
    return diff;
}

std::pair<std::vector<uint64_t>, uint64_t> bigNumDivide(const std::vector<uint64_t> &a, uint64_t divisor) {
    std::vector<uint64_t> quotient(a.size(), 0ULL);
    uint64_t remainder = 0ULL;
    for (int i = (int)a.size() - 1; i >= 0; --i) {
        __uint128_t temp = ((__uint128_t)remainder << 64) | a[i];
        uint64_t q = (uint64_t)(temp / divisor);
        uint64_t r = (uint64_t)(temp % divisor);
        quotient[i] = q;
        remainder = r;
    }
    while (!quotient.empty() && quotient.back() == 0ULL)
        quotient.pop_back();
    return {quotient, remainder};
}

long double hexStrToLongDouble(const std::string &hex) {
    long double result = 0.0L;
    for (char c : hex) {
        result *= 16.0L;
        if (c >= '0' && c <= '9')
            result += (c - '0');
        else if (c >= 'a' && c <= 'f')
            result += (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            result += (c - 'A' + 10);
    }
    return result;
}

//------------------------------------------------------------------------------
static inline std::string padHexTo64(const std::string &hex) {
    return (hex.size() >= 64) ? hex : std::string(64 - hex.size(), '0') + hex;
}

static inline Int hexToInt(const std::string &hex) {
    Int number;
    char buf[65] = {0};
    std::strncpy(buf, hex.c_str(), 64);
    number.SetBase16(buf);
    return number;
}

static inline std::string intToHex(const Int &value) {
    Int temp;
    temp.Set((Int *)&value);
    return temp.GetBase16();
}

static inline bool intGreater(const Int &a, const Int &b) {
    std::string ha = ((Int &)a).GetBase16();
    std::string hb = ((Int &)b).GetBase16();
    if (ha.size() != hb.size()) return (ha.size() > hb.size());
    return (ha > hb);
}

static inline bool isEven(const Int &number) {
    return ((Int &)number).IsEven();
}

static inline std::string intXToHex64(const Int &x) {
    Int temp;
    temp.Set((Int *)&x);
    std::string hex = temp.GetBase16();
    if (hex.size() < 64)
        hex.insert(0, 64 - hex.size(), '0');
    return hex;
}

static inline std::string pointToCompressedHex(const Point &point) {
    return (isEven(point.y) ? "02" : "03") + intXToHex64(point.x);
}

static inline void pointToCompressedBin(const Point &point, uint8_t outCompressed[33]) {
    outCompressed[0] = isEven(point.y) ? 0x02 : 0x03;
    Int temp;
    temp.Set((Int *)&point.x);
    for (int i = 0; i < 32; i++) {
        outCompressed[1 + i] = (uint8_t)temp.GetByte(31 - i);
    }
}

//------------------------------------------------------------------------------
inline void prepareShaBlock(const uint8_t *dataSrc, size_t dataLen, uint8_t *outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, dataLen);
    outBlock[dataLen] = 0x80;
    const uint32_t bitLen = (uint32_t)(dataLen * 8);
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >> 8) & 0xFF);
    outBlock[63] = (uint8_t)(bitLen & 0xFF);
}

inline void prepareRipemdBlock(const uint8_t *dataSrc, uint8_t *outBlock) {
    std::fill_n(outBlock, 64, 0);
    std::memcpy(outBlock, dataSrc, 32);
    outBlock[32] = 0x80;
    const uint32_t bitLen = 256;
    outBlock[60] = (uint8_t)((bitLen >> 24) & 0xFF);
    outBlock[61] = (uint8_t)((bitLen >> 16) & 0xFF);
    outBlock[62] = (uint8_t)((bitLen >> 8) & 0xFF);
    outBlock[63] = (uint8_t)(bitLen & 0xFF);
}

// Computing hash160 using avx2 (8 hashes per try)
static void computeHash160BatchBinSingle(int numKeys, uint8_t pubKeys[][33], uint8_t hashResults[][20]) {
    alignas(32) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
    alignas(32) std::array<std::array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
    alignas(32) std::array<std::array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;
    alignas(32) std::array<std::array<uint8_t, 20>, HASH_BATCH_SIZE> ripemdOutputs;

    const size_t totalBatches = (numKeys + (HASH_BATCH_SIZE - 1)) / HASH_BATCH_SIZE;
    for (size_t batch = 0; batch < totalBatches; batch++) {
        const size_t batchCount = std::min<size_t>(HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);
        for (size_t i = 0; i < batchCount; i++) {
            const size_t idx = batch * HASH_BATCH_SIZE + i;
            prepareShaBlock(pubKeys[idx], 33, shaInputs[i].data());
        }
        for (size_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
            std::memcpy(shaInputs[i].data(), shaInputs[0].data(), 64);
        }
        const uint8_t *inPtr[HASH_BATCH_SIZE];
        uint8_t *outPtr[HASH_BATCH_SIZE];
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i] = shaInputs[i].data();
            outPtr[i] = shaOutputs[i].data();
        }
        // SHA256 (avx2)
        sha256avx2_8B(inPtr[0], inPtr[1], inPtr[2], inPtr[3],
                      inPtr[4], inPtr[5], inPtr[6], inPtr[7],
                      outPtr[0], outPtr[1], outPtr[2], outPtr[3],
                      outPtr[4], outPtr[5], outPtr[6], outPtr[7]);

        // Preparing Ripemd160
        for (size_t i = 0; i < batchCount; i++) {
            prepareRipemdBlock(shaOutputs[i].data(), ripemdInputs[i].data());
        }
        for (size_t i = batchCount; i < HASH_BATCH_SIZE; i++) {
            std::memcpy(ripemdInputs[i].data(), ripemdInputs[0].data(), 64);
        }
        for (int i = 0; i < HASH_BATCH_SIZE; i++) {
            inPtr[i] = ripemdInputs[i].data();
            outPtr[i] = ripemdOutputs[i].data();
        }
        // Ripemd160 (avx2)
        ripemd160avx2::ripemd160avx2_32(
            (unsigned char *)inPtr[0],
            (unsigned char *)inPtr[1],
            (unsigned char *)inPtr[2],
            (unsigned char *)inPtr[3],
            (unsigned char *)inPtr[4],
            (unsigned char *)inPtr[5],
            (unsigned char *)inPtr[6],
            (unsigned char *)inPtr[7],
            outPtr[0], outPtr[1], outPtr[2], outPtr[3],
            outPtr[4], outPtr[5], outPtr[6], outPtr[7]
        );
        for (size_t i = 0; i < batchCount; i++) {
            const size_t idx = batch * HASH_BATCH_SIZE + i;
            std::memcpy(hashResults[idx], ripemdOutputs[i].data(), 20);
        }
    }
}

//------------------------------------------------------------------------------
static void printUsage(const char *programName) {
    std::cerr << "Usage: " << programName << " -t <number_of_threads> -h <hash160_hex> [-p <puzzle> | -r <startHex:endHex>] -b <prefix_length> [-R | -S] \n";
    std::cerr << "  -R : Use random mode (default is sequential)\n";
    std::cerr << "  -S : Use sequential mode\n";
}

static std::string formatElapsedTime(double seconds) {
    int hrs = (int)seconds / 3600;
    int mins = ((int)seconds % 3600) / 60;
    int secs = (int)seconds % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hrs << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

//------------------------------------------------------------------------------
// Function to generate partial match information
std::string generatePartialMatchInfo(const std::string& privateKeyHex, const std::string& publicKeyHex,
                                    const uint8_t* foundHash160, const uint8_t* targetHash160, int prefixLength) {
    std::ostringstream oss;
    oss << "================== PARTIAL MATCH FOUND! ============\n";
    oss << "Prefix length : " << prefixLength << " bytes\n";
    oss << "Private Key   : " << privateKeyHex << "\n";
    oss << "Public Key    : " << publicKeyHex << "\n";
    oss << "Found Hash160 : ";
    for (int b = 0; b < 20; b++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)foundHash160[b];
    }
    oss << "\n";
    oss << "Target Hash160: ";
    for (int b = 0; b < 20; b++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)targetHash160[b];
    }
    oss << "\n";
    oss << "Matched bytes : ";
    for (int b = 0; b < prefixLength; b++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)targetHash160[b];
    }
    oss << std::endl;
    return oss.str();
}

// Function to print status block and partial match information
static void printStatsBlock(int maxt, int numCPUs, const std::string &targetHash160Hex,
                            const std::string &rangeStr, double mkeysPerSec,
                            unsigned long long totalChecked, double elapsedTime,
                            int puzzle, bool randomMode, const std::string& partialMatchInfo = "") {
    std::lock_guard<std::mutex> lock(coutMutex);

    // Move cursor to the top-left corner
    std::cout << "\033[2J\033[1;1H\r";

    // Print the status block (lines 1-8)
    std::cout << "================= WORK IN PROGRESS =================\n";
    std::cout << "Puzzle        : " << puzzle << "\n"; // Print puzzle value
    std::cout << "Mode          : " << (randomMode ? "Random" : "Sequential") << "\n"; // Add Mode
    std::cout << "Range         : " << rangeStr << "\n";
    std::cout << "Target Hash160: " << targetHash160Hex << "\n";
    std::cout << "CPU Threads   : " << numCPUs << "\n";
	std::cout << "Alloc Threads : " << maxt << "\n";
    std::cout << "Mkeys/s       : " << std::fixed << std::setprecision(2) << mkeysPerSec << "\n";
    std::cout << "Total Checked : " << totalChecked << "\n";
    std::cout << "Elapsed Time  : " << formatElapsedTime(elapsedTime) << "\n";

    // Print the partial match information (lines 11-16)
    if (!partialMatchInfo.empty()) {
        // Move cursor to line 11
        std::cout << "\033[10;1H";
        // Clear the line
        std::cout << "\033[K";
        std::cout << partialMatchInfo;

        // Save partial match to MATCH.txt
        std::ofstream matchFile("MATCH.txt", std::ios::app);
        if (matchFile) {
            matchFile << partialMatchInfo << "\n";
        } else {
            std::cerr << "Cannot open MATCH.txt for writing\n";
        }
    }

    std::cout.flush();
}

//------------------------------------------------------------------------------
struct ThreadRange {
    std::string startHex;
    std::string endHex;
};

static std::vector<ThreadRange> g_threadRanges;

class Timer {
public:
    static std::string getSeed(int length) {
        auto now = std::chrono::high_resolution_clock::now();
        auto epoch = now.time_since_epoch();
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
        std::ostringstream oss;
        oss << std::hex << value;
        return oss.str().substr(0, length);
    }
};

class Xoshiro256plus {
public:
    Xoshiro256plus(uint64_t seed = 0) {
        state[0] = seed;
        for (int i = 1; i < 4; ++i) {
            state[i] = 1812433253ULL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
        }
    }

    uint64_t next() {
        const uint64_t result = state[0] + state[3];
        const uint64_t t = state[1] << 17;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 45);

        return result;
    }

private:
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    std::array<uint64_t, 4> state;
};

Int generateRandomPrivateKey(Int minKey, Int range, Xoshiro256plus &rng) {
    Int randomPrivateKey((uint64_t)0);

    // Generate random values in chunks of 64 bits using Xoshiro256plus
    for (int i = 0; i < NB64BLOCK; ++i) {
        uint64_t randVal = rng.next();
        randomPrivateKey.ShiftL(64); // Shift left by 64 bits
        randomPrivateKey.Add(randVal);
    }

    // Apply modulo operation and add minKey
    randomPrivateKey.Mod(&range);
    randomPrivateKey.Add(&minKey);

    return randomPrivateKey;
}

Int minKey, maxKey;
	int maxt = 1; //Declare the max threads allowed. Minimum is one. Range checked for 0 and true max detected at parameters passed -dev_nullish
int main(int argc, char *argv[]) {
    bool hash160Provided = false, rangeProvided = false, puzzleProvided = false;
    bool randomMode = false; // Default to sequential mode
    std::string targetHash160Hex;
    std::vector<uint8_t> targetHash160;
    int puzzle = 0; // Declare puzzle variable
	int maxt = 1; //Declare the max threads allowed. Minimum is one. Range checked at parameters passed. Should never be 0. -dev_nullish
    std::string rangeStartHex, rangeEndHex;
    // Parse command-line arguments
	int numCPUs = omp_get_num_procs(); // Removed const since the number of threads/cpus can vary from max collected here by omp_get_num_procs() - dev_nullish
	for (int i = 1; i < argc; i++) {
	if (!std::strcmp(argv[i], "-t") && i + 1 < argc) {
            maxt = std::stoi(argv[++i]);
// Range check for the number of threads versus maximum, non zero and up to equal of total threads possible discovered by numCPUs - dev_nullish
            if (((maxt > numCPUs) | (maxt<1))) {
                std::cerr << "Invalid thread size or be larger than 0. Also check your WSL2 settings for number of threads if using that defined as lower.\n";
                return 1;
			}
		} else if (!std::strcmp(argv[i], "-h") && i + 1 < argc) { // Use -h for hash160_hex
            targetHash160Hex = argv[++i];
            hash160Provided = true;
            // Convert the hex string to a byte array
            targetHash160.resize(20);
            for (size_t j = 0; j < 20; j++) {
                targetHash160[j] = std::stoul(targetHash160Hex.substr(j * 2, 2), nullptr, 16);
            }
        } else if (!std::strcmp(argv[i], "-p") && i + 1 < argc) {
            puzzle = std::stoi(argv[++i]);
            if (puzzle <= 0) {
                std::cerr << "Invalid puzzle value. Must be greater than 0.\n";
                return 1;
            }
            puzzleProvided = true;
        } else if (!std::strcmp(argv[i], "-r") && i + 1 < argc) {
            std::string range = argv[++i];
            size_t colonPos = range.find(':');
            if (colonPos == std::string::npos) {
                std::cerr << "Invalid range format. Expected startHex:endHex.\n";
                return 1;
            }
            rangeStartHex = range.substr(0, colonPos);
            rangeEndHex = range.substr(colonPos + 1);
            rangeProvided = true;
        } else if (!std::strcmp(argv[i], "-b") && i + 1 < argc) {
            g_prefixLength = std::stoi(argv[++i]);
            if (g_prefixLength <= 0 || g_prefixLength > 20) {
                std::cerr << "Invalid prefix length. Must be between 1 and 20.\n";
                return 1;
            }
        } else if (!std::strcmp(argv[i], "-R")) {
            randomMode = true; // Enable random mode
        } else if (!std::strcmp(argv[i], "-S")) {
            randomMode = false; // Enable sequential mode
        } else {
            std::cerr << "Unknown parameter: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
		}	
    }

    if (!hash160Provided || (!rangeProvided && !puzzleProvided)) {
        std::cerr << "Both -h and (-p or -r) are required!\n";
        printUsage(argv[0]);
        return 1;
    }

    if (puzzleProvided) {
        // Calculate the range based on the puzzle number
        Int one;
        one.SetBase10(const_cast<char *>("1"));
        minKey = one;
        minKey.ShiftL(puzzle - 1); // Start of range: 2^(puzzle-1)
        maxKey = one;
        maxKey.ShiftL(puzzle);     // End of range: 2^puzzle
        maxKey.Sub(&one);          // End of range: 2^puzzle - 1

        // Convert minKey and maxKey to hex strings for rangeStartHex and rangeEndHex
        rangeStartHex = intToHex(minKey);
        rangeEndHex = intToHex(maxKey);
    } else if (rangeProvided) {
        // Calculate puzzle number based on the range
        Int rangeStart = hexToInt(rangeStartHex);
        Int rangeEnd = hexToInt(rangeEndHex);
        Int rangeSize;
        rangeSize.Sub(&rangeEnd, &rangeStart);
        rangeSize.AddOne();

        // Calculate the number of bits required to represent the range size
        int bits = 0;
        Int temp;
        temp.Set(&rangeSize);
        while (!temp.IsZero()) {
            temp.ShiftR(1);
            bits++;
        }
        puzzle = bits;
    }

    // Convert range to big numbers
    auto rangeStart = hexToBigNum(rangeStartHex);
    auto rangeEnd = hexToBigNum(rangeEndHex);

    // Validate range
    bool validRange = false;
    if (rangeStart.size() < rangeEnd.size()) {
        validRange = true;
    } else if (rangeStart.size() > rangeEnd.size()) {
        validRange = false;
    } else {
        validRange = true;
        for (int i = (int)rangeStart.size() - 1; i >= 0; --i) {
            if (rangeStart[i] < rangeEnd[i]) {
                break;
            } else if (rangeStart[i] > rangeEnd[i]) {
                validRange = false;
                break;
            }
        }
    }
    if (!validRange) {
        std::cerr << "Range start must be less than range end.\n";
        return 1;
    }

    auto rangeSize = bigNumSubtract(rangeEnd, rangeStart);
    rangeSize = bigNumAdd(rangeSize, singleElementVector(1ULL));

    const std::string rangeSizeHex = bigNumToHex(rangeSize);

    const long double totalRangeLD = hexStrToLongDouble(rangeSizeHex);
    
    g_threadPrivateKeys.resize(numCPUs, "0");

    auto [chunkSize, remainder] = bigNumDivide(rangeSize, (uint64_t)numCPUs);
    g_threadRanges.resize(numCPUs);

    std::vector<uint64_t> currentStart = rangeStart;
    for (int t = 0; t < numCPUs; t++) {
        auto currentEnd = bigNumAdd(currentStart, chunkSize);
        if (t < (int)remainder) {
            currentEnd = bigNumAdd(currentEnd, singleElementVector(1ULL));
        }
        currentEnd = bigNumSubtract(currentEnd, singleElementVector(1ULL));

        g_threadRanges[t].startHex = bigNumToHex(currentStart);
        g_threadRanges[t].endHex = bigNumToHex(currentEnd);

        currentStart = bigNumAdd(currentEnd, singleElementVector(1ULL));
    }
    const std::string displayRange = g_threadRanges.front().startHex + ":" + g_threadRanges.back().endHex;

    unsigned long long globalComparedCount = 0ULL;
    double globalElapsedTime = 0.0;
    double mkeysPerSec = 0.0;

    const auto tStart = std::chrono::high_resolution_clock::now();
    auto lastStatusTime = tStart;
    auto lastSaveTime = tStart;

    bool matchFound = false;
    std::string foundPrivateKeyHex;
    std::string foundPublicKeyHex; // Declare foundPublicKeyHex

    Int one;
    one.SetBase10(const_cast<char *>("1"));
    Int minKey = one;
    minKey.ShiftL(puzzle - 1); // Start of range: 2^(puzzle-1)
    Int maxKey = one;
    maxKey.ShiftL(puzzle); // End of range: 2^puzzle - 1
    maxKey.Sub(&one);
    Int range = maxKey;
    range.Sub(&minKey);

    Secp256K1 secp;
    secp.Init();

    // PARRALEL COMPUTING BLOCK
#pragma omp parallel num_threads(numCPUs) \
    shared(globalComparedCount, globalElapsedTime, mkeysPerSec, matchFound, \
           foundPrivateKeyHex, foundPublicKeyHex, lastStatusTime, lastSaveTime, g_progressSaveCount, \
           g_threadPrivateKeys)
    {
        const int threadId = omp_get_thread_num();

        // Initialize Xoshiro256plus PRNG for this thread
        Xoshiro256plus rng(std::chrono::steady_clock::now().time_since_epoch().count() + threadId);

        Int privateKey = hexToInt(g_threadRanges[threadId].startHex);
        const Int threadRangeEnd = hexToInt(g_threadRanges[threadId].endHex);

#pragma omp critical
        {
            g_threadPrivateKeys[threadId] = padHexTo64(intToHex(privateKey));
        }

        // Precomputing +i*G and -i*G for i=0..255
        std::vector<Point> plusPoints(POINTS_BATCH_SIZE);
        std::vector<Point> minusPoints(POINTS_BATCH_SIZE);
        for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
            Int tmp;
            tmp.SetInt32(i);
            Point p = secp.ComputePublicKey(&tmp);
            plusPoints[i] = p;
            p.y.ModNeg();
            minusPoints[i] = p;
        }

        // Arrays for batch-adding
        std::vector<Int> deltaX(POINTS_BATCH_SIZE);
        IntGroup modGroup(POINTS_BATCH_SIZE);

        // Save 512 publickeys
        const int fullBatchSize = 2 * POINTS_BATCH_SIZE;
        std::vector<Point> pointBatch(fullBatchSize);

        // Buffers for hashing
        uint8_t localPubKeys[fullBatchSize][33];
        uint8_t localHashResults[HASH_BATCH_SIZE][20];
        int localBatchCount = 0;
        int pointIndices[HASH_BATCH_SIZE];

        // Local count
        unsigned long long localComparedCount = 0ULL;

        // Download the target (hash160) в __m128i for fast compare
        __m128i target16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(targetHash160.data()));

        // Main loop for generating random private keys
        while (!matchFound) {
            Int currentBatchKey;
            if (randomMode) {
                // Generate a random private key within the thread's range using Xoshiro256plus
                currentBatchKey = generateRandomPrivateKey(minKey, range, rng);
            } else {
                // Sequential mode
                if (intGreater(privateKey, threadRangeEnd)) {
                    break;
                }
                currentBatchKey.Set(&privateKey);
            }

            Point startPoint = secp.ComputePublicKey(&currentBatchKey);

#pragma omp critical
            {
                g_threadPrivateKeys[threadId] = padHexTo64(intToHex(privateKey));
            }

            // Divide the batch of 512 keys into 2 blocks of 256 keys, count +256 and -256 from the center G-point of the batch
            // First pointBatch[0..255] +
            for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
                deltaX[i].ModSub(&plusPoints[i].x, &startPoint.x);
            }
            modGroup.Set(deltaX.data());
            modGroup.ModInv();
            for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
                Point tempPoint = startPoint;
                Int deltaY;
                deltaY.ModSub(&plusPoints[i].y, &startPoint.y);
                Int slope;
                slope.ModMulK1(&deltaY, &deltaX[i]);
                Int slopeSq;
                slopeSq.ModSquareK1(&slope);

                Int tmpX;
                tmpX.Set(&startPoint.x);
                tmpX.ModNeg();
                tmpX.ModAdd(&slopeSq);
                tmpX.ModSub(&plusPoints[i].x);
                tempPoint.x.Set(&tmpX);

                Int diffX;
                diffX.Set(&startPoint.x);
                diffX.ModSub(&tempPoint.x);
                diffX.ModMulK1(&slope);
                tempPoint.y.ModNeg();
                tempPoint.y.ModAdd(&diffX);

                pointBatch[i] = tempPoint;
            }

            // Second pointBatch[256..511] -
            for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
                Point tempPoint = startPoint;
                Int deltaY;
                deltaY.ModSub(&minusPoints[i].y, &startPoint.y);
                Int slope;
                slope.ModMulK1(&deltaY, &deltaX[i]);
                Int slopeSq;
                slopeSq.ModSquareK1(&slope);

                Int tmpX;
                tmpX.Set(&startPoint.x);
                tmpX.ModNeg();
                tmpX.ModAdd(&slopeSq);
                tmpX.ModSub(&minusPoints[i].x);
                tempPoint.x.Set(&tmpX);

                Int diffX;
                diffX.Set(&startPoint.x);
                diffX.ModSub(&tempPoint.x);
                diffX.ModMulK1(&slope);
                tempPoint.y.ModNeg();
                tempPoint.y.ModAdd(&diffX);

                pointBatch[POINTS_BATCH_SIZE + i] = tempPoint;
            }

            // Construct local buffer
            for (int i = 0; i < fullBatchSize; i++) {
                pointToCompressedBin(pointBatch[i], localPubKeys[localBatchCount]);
                pointIndices[localBatchCount] = i;
                localBatchCount++;

                // 8 keys are ready - time to use avx2
                if (localBatchCount == HASH_BATCH_SIZE) {
                    computeHash160BatchBinSingle(localBatchCount, localPubKeys, localHashResults);
                    // Results check
                    for (int j = 0; j < HASH_BATCH_SIZE; j++) {
                        __m128i cand16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(localHashResults[j]));
                        __m128i cmp = _mm_cmpeq_epi8(cand16, target16);
                        uint16_t bitmask = (0xFFFF >> (16 - 4 * g_prefixLength)) << (16 - 4 * g_prefixLength);

                        // Check for full match
                        if ((_mm_movemask_epi8(cmp) & bitmask) == bitmask) {
                            if (!matchFound && std::memcmp(localHashResults[j], targetHash160.data(), 20) == 0) {
#pragma omp critical
                                {
                                    if (!matchFound) {
                                        matchFound = true;
                                        auto tEndTime = std::chrono::high_resolution_clock::now();
                                        globalElapsedTime = std::chrono::duration<double>(tEndTime - tStart).count();
                                        mkeysPerSec = (double)(globalComparedCount + localComparedCount) / globalElapsedTime / 1e6;

                                        // Recovering private key
                                        Int matchingPrivateKey;
                                        matchingPrivateKey.Set(&currentBatchKey);
                                        int idx = pointIndices[j];
                                        if (idx < 256) {
                                            Int offset;
                                            offset.SetInt32(idx);
                                            matchingPrivateKey.Add(&offset);
                                        } else {
                                            Int offset;
                                            offset.SetInt32(idx - 256);
                                            matchingPrivateKey.Sub(&offset);
                                        }
                                        foundPrivateKeyHex = padHexTo64(intToHex(matchingPrivateKey));
                                        Point matchedPoint = pointBatch[idx];
                                        foundPublicKeyHex = pointToCompressedHex(matchedPoint); // Assign foundPublicKeyHex
                                        // Save full match to KEYFOUND.txt
                                        std::ofstream keyFoundFile("KEYFOUND.txt", std::ios::app);
                                        if (keyFoundFile) {
                                        keyFoundFile << "==================  FULL MATCH FOUND! ==================\n";
                                        keyFoundFile << "Private Key   : " << foundPrivateKeyHex << "\n";
                                        keyFoundFile << "Public Key    : " << foundPublicKeyHex << "\n";
                                        keyFoundFile << "Total Checked : " << globalComparedCount << "\n";
                                        keyFoundFile << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
                                        keyFoundFile << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
                                        } else {
                                        std::cerr << "Cannot open KEYFOUND.txt for writing\n";
                                       }
                                    }
                                }
#pragma omp cancel parallel
                            }
                            localComparedCount++;
                        }

                        // Check for partial match (independent of full match)
                        bool bytesMatch = true;
                        for (int b = 0; b < g_prefixLength; b++) {
                            if (localHashResults[j][b] != targetHash160.data()[b]) {
                                bytesMatch = false;
                                break;
                            }
                        }
                        if (bytesMatch) {
                            // Recovering private key for partial match
                            Int matchingPrivateKey;
                            matchingPrivateKey.Set(&currentBatchKey);
                            int idx = pointIndices[j];
                            if (idx < 256) {
                                Int offset;
                                offset.SetInt32(idx);
                                matchingPrivateKey.Add(&offset);
                            } else {
                                Int offset;
                                offset.SetInt32(idx - 256);
                                matchingPrivateKey.Sub(&offset);
                            }
                            foundPrivateKeyHex = padHexTo64(intToHex(matchingPrivateKey));
                            Point matchedPoint = pointBatch[idx];
                            foundPublicKeyHex = pointToCompressedHex(matchedPoint); // Assign foundPublicKeyHex

                            // Generate partial match information
                            std::string partialMatchInfo = generatePartialMatchInfo(foundPrivateKeyHex, foundPublicKeyHex,
                                                                                   localHashResults[j], targetHash160.data(), g_prefixLength);

                            // Print status block and partial match information
printStatsBlock(maxt, numCPUs, targetHash160Hex, displayRange,
                mkeysPerSec, globalComparedCount,
                globalElapsedTime, puzzle, randomMode, partialMatchInfo);
                        }
                        localComparedCount++;
                    }
                    localBatchCount = 0;
                }
            }

            if (!randomMode) {
                // Increment private key for sequential mode
                Int step;
                step.SetInt32(fullBatchSize - 2); // 510
                privateKey.Add(&step);
            }

            // Time to show status
            auto now = std::chrono::high_resolution_clock::now();
            double secondsSinceStatus = std::chrono::duration<double>(now - lastStatusTime).count();
            if (secondsSinceStatus >= statusIntervalSec) {
#pragma omp critical
                {
                    globalComparedCount += localComparedCount;
                    localComparedCount = 0ULL;
                    globalElapsedTime = std::chrono::duration<double>(now - tStart).count();
                    mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;

                    // Print status block without partial match information
                    printStatsBlock(maxt, numCPUs, targetHash160Hex, displayRange,
                                   mkeysPerSec, globalComparedCount,
                                   globalElapsedTime, puzzle, randomMode);
                    lastStatusTime = now;
                }
            }

            if (matchFound) {
                break;
            }
        } // while(true)

        // Adding local count
#pragma omp atomic
        globalComparedCount += localComparedCount;
    } // end of parallel section

    // Main results
    auto tEnd = std::chrono::high_resolution_clock::now();
    globalElapsedTime = std::chrono::duration<double>(tEnd - tStart).count();

    if (!matchFound) {
        mkeysPerSec = (double)globalComparedCount / globalElapsedTime / 1e6;
        std::cout << "\nNo match found.\n";
        std::cout << "Total Checked : " << globalComparedCount << "\n";
        std::cout << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
        std::cout << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
        return 0;
    }

    // If the key was found
    std::cout << "==================  FULL MATCH FOUND! ==================\n";
    std::cout << "Private Key   : " << foundPrivateKeyHex << "\n";
    std::cout << "Total Checked : " << globalComparedCount << "\n";
    std::cout << "Elapsed Time  : " << formatElapsedTime(globalElapsedTime) << "\n";
    std::cout << "Speed         : " << mkeysPerSec << " Mkeys/s\n";
    return 0;
}
