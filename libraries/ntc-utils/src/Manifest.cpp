/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <ntc-utils/Manifest.h>
#include <filesystem>
#include <json/value.h>
#include <json/reader.h>
#include <algorithm>

namespace fs = std::filesystem;

void LowercaseString(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](uint8_t ch) { return std::tolower(ch); });
}

void UppercaseString(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](uint8_t ch) { return std::toupper(ch); });
}

bool IsSupportedImageFileExtension(std::string const& extension)
{
    return extension == ".png" ||
           extension == ".jpg" ||
           extension == ".jpeg" ||
           extension == ".tga" ||
           extension == ".exr";
}

static void ComputeDistinctImageNames(Manifest& manifest)
{
    std::string commonName;

    bool isFirstImage = true;
    for (ManifestEntry& entry : manifest.textures)
    {
        // Detect a common prefix for all file names
        if (isFirstImage)
        {
            commonName = entry.entryName;
            isFirstImage = false;
        }
        else
        {
            size_t i;
            for (i = 0; i < commonName.size() && i < entry.entryName.size(); ++i)
            {
                if (tolower(commonName[i]) != tolower(entry.entryName[i]))
                    break;
            }
            commonName.resize(i);
        }
    }

    if (commonName.empty())
        return;

    for (ManifestEntry& entry : manifest.textures)
    {
        std::string distinctName = entry.entryName.substr(commonName.size());
        if (!distinctName.empty())
        {
            distinctName[0] = char(toupper(distinctName[0]));
            entry.entryName = distinctName;
        }
    }
}

void GenerateManifestFromDirectory(const char* path, bool loadMips, Manifest& outManifest)
{
    for (const fs::directory_entry& directoryEntry : fs::directory_iterator(path))
    {
        const fs::path& fileName = directoryEntry.path();

        // Get a lowercase file extension for case-insensitive comparison
        std::string extension = fileName.extension().generic_string();
        LowercaseString(extension);

        if (!IsSupportedImageFileExtension(extension))
            continue;

        ManifestEntry& entry = outManifest.textures.emplace_back();
        entry.fileName = fileName.generic_string();
        entry.entryName = fileName.stem().generic_string();
        entry.mipLevel = 0;
    }

    if (loadMips)
    {
        for (const fs::directory_entry& directoryEntry : fs::directory_iterator(fs::path(path) / "mips"))
        {
            const fs::path& fileName = directoryEntry.path();

            // Get a lowercase file extension for case-insensitive comparison
            std::string extension = fileName.extension().generic_string();
            LowercaseString(extension);

            if (extension != ".png" && extension != ".jpg" && extension != ".tga" && extension != ".exr")
                continue;

            // Parse the file name, assuming it follows this pattern: <name>.<mip>.<type>
            fs::path mip = fileName.stem().extension();
            fs::path name = fileName.stem().stem();

            if (mip.empty() || name.empty())
                continue;

            auto found = std::find_if(outManifest.textures.begin(), outManifest.textures.end(),
                [&name](const ManifestEntry& entry) { return entry.entryName == name; });

            if (found == outManifest.textures.end())
                continue;
            
            int mipLevel = 0;
            if (sscanf(mip.generic_string().c_str(), ".%d", &mipLevel) != 1)
                continue;

            if (mipLevel >= NTC_MAX_MIPS)
                continue;
            
            ManifestEntry& entry = outManifest.textures.emplace_back();
            entry.fileName = fileName.generic_string();
            entry.entryName = name.generic_string();
            entry.mipLevel = mipLevel;
        }
    }

    ComputeDistinctImageNames(outManifest);
}

void GenerateManifestFromFileList(std::vector<const char *> const &files, Manifest &outManifest)
{
    for (char const* name : files)
    {
        const fs::path& fileName = name;

        ManifestEntry& entry = outManifest.textures.emplace_back();
        entry.fileName = fileName.generic_string();
        entry.entryName = fileName.stem().generic_string();
        entry.mipLevel = 0;
    }
    
    ComputeDistinctImageNames(outManifest);
}

static bool ReadFileIntoVector(FILE* inputFile, std::vector<char>& vector)
{
    if (fseek(inputFile, 0, SEEK_END))
        return false;
    long fileSize = ftell(inputFile);
    if (fseek(inputFile, 0, SEEK_SET))
        return false;
    vector.resize(fileSize);
    if (fread(vector.data(), fileSize, 1, inputFile) != 1)
        return false;
    return true;
}

std::optional<ntc::BlockCompressedFormat> ParseBlockCompressedFormat(char const* format, bool enableAuto)
{
    if (!format || !format[0])
        return ntc::BlockCompressedFormat::None;

    std::string uppercaseFormat = format;
    UppercaseString(uppercaseFormat);

    if (uppercaseFormat == "NONE")
        return ntc::BlockCompressedFormat::None;
    if (uppercaseFormat == "BC1")
        return ntc::BlockCompressedFormat::BC1;
    if (uppercaseFormat == "BC2")
        return ntc::BlockCompressedFormat::BC2;
    if (uppercaseFormat == "BC3")
        return ntc::BlockCompressedFormat::BC3;
    if (uppercaseFormat == "BC4")
        return ntc::BlockCompressedFormat::BC4;
    if (uppercaseFormat == "BC5")
        return ntc::BlockCompressedFormat::BC5;
    if (uppercaseFormat == "BC6" || uppercaseFormat == "BC6H")
        return ntc::BlockCompressedFormat::BC6;
    if (uppercaseFormat == "BC7")
        return ntc::BlockCompressedFormat::BC7;
    if (uppercaseFormat == "AUTO" && enableAuto)
        return BlockCompressedFormat_Auto;

    return std::optional<ntc::BlockCompressedFormat>();
}

SemanticLabel ParseSemanticLabel(const char* label)
{
    std::string uppercaseLabel = label;
    UppercaseString(uppercaseLabel);

    if (uppercaseLabel == "ALBEDO")
        return SemanticLabel::Albedo;
    if (uppercaseLabel == "ALPHA" || uppercaseLabel == "MASK" || uppercaseLabel == "ALPHAMASK")
        return SemanticLabel::AlphaMask;
    if (uppercaseLabel == "DISPL" || uppercaseLabel == "DISPLACEMENT")
        return SemanticLabel::Displacement;
    if (uppercaseLabel == "EMISSIVE" || uppercaseLabel == "EMISSION")
        return SemanticLabel::Emissive;
    if (uppercaseLabel == "METALNESS" || uppercaseLabel == "METALLIC")
        return SemanticLabel::Metalness;
    if (uppercaseLabel == "NORMAL")
        return SemanticLabel::Normal;
    if (uppercaseLabel == "OCCLUSION" || uppercaseLabel == "AO")
        return SemanticLabel::Occlusion;
    if (uppercaseLabel == "ROUGHNESS")
        return SemanticLabel::Roughness;
    if (uppercaseLabel == "TRANSMISSION")
        return SemanticLabel::Transmission;
    if (uppercaseLabel == "SPECULARCOLOR")
        return SemanticLabel::SpecularColor;
    if (uppercaseLabel == "GLOSSINESS")
        return SemanticLabel::Glossiness;

    return SemanticLabel::None;
}

char const* SemanticLabelToString(SemanticLabel label)
{
    switch (label)
    {
        case SemanticLabel::None:
            return "(None)";
        case SemanticLabel::Albedo:
            return "Albedo";
        case SemanticLabel::AlphaMask:
            return "AlphaMask";
        case SemanticLabel::Displacement:
            return "Displacement";
        case SemanticLabel::Emissive:
            return "Emissive";
        case SemanticLabel::Glossiness:
            return "Glossiness";
        case SemanticLabel::Metalness:
            return "Metalness";
        case SemanticLabel::Normal:
            return "Normal";
        case SemanticLabel::Occlusion:
            return "Occlusion";
        case SemanticLabel::Roughness:
            return "Roughness";
        case SemanticLabel::SpecularColor:
            return "SpecularColor";
        case SemanticLabel::Transmission:
            return "Transmission";
        default:
            static char string[16];
            snprintf(string, sizeof(string), "%d", int(label));
            return string;
    }
}

int GetSemanticChannelCount(SemanticLabel label)
{
    switch (label)
    {
        case SemanticLabel::Albedo:
        case SemanticLabel::Emissive:
        case SemanticLabel::Normal:
        case SemanticLabel::SpecularColor:
            return 3;

        case SemanticLabel::AlphaMask:
        case SemanticLabel::Displacement:
        case SemanticLabel::Glossiness:
        case SemanticLabel::Metalness:
        case SemanticLabel::Occlusion:
        case SemanticLabel::Roughness:
        case SemanticLabel::Transmission:
            return 1;

        default:
            return 0;
    }
}

bool ReadManifestFromFile(const char* fileName, Manifest& outManifest, std::string& outError)
{
    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::CharReader* reader = builder.newCharReader();

    FILE* inputFile = fopen(fileName, "rb");
    if (!inputFile)
    {
        std::ostringstream oss;
        oss << "Cannot open manifest file '" << fileName << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    std::vector<char> fileContents;
    bool success = ReadFileIntoVector(inputFile, fileContents);
    fclose(inputFile);
    
    if (!success)
    {
        std::ostringstream oss;
        oss << "Error while reading manifest file '" << fileName << "': " << strerror(errno);
        outError = oss.str();
        return false;
    }

    Json::Value root;
    Json::String errorMessages;
    if (!reader->parse(fileContents.data(), fileContents.data() + fileContents.size(), &root, &errorMessages))
    {
        std::ostringstream oss;
        oss << "Cannot parse manifest file '" << fileName << "': " << errorMessages;
        outError = oss.str();
        return false;
    }

    if (!root.isObject() && !root.isArray())
    {
        outError = "Malformed manifest: document root must be an object or an array.";
        return false;
    }

    // Select between the new format `{ "textures": [...] }` and the old format `[...]` for compatibility.
    // The old format will be removed someday.
    Json::Value const& textures = root.isObject() ? root["textures"] : root;
    if (!textures.isArray() || textures.empty())
    {
        outError = "Malformed manifest: must contain a non-empty 'textures' array.";
        return false;
    }

    if (root.isObject())
    {
        if (root["width"].isNumeric())
            outManifest.width = root["width"].asInt();
        if (root["height"].isNumeric())
            outManifest.height = root["height"].asInt();
    }

    fs::path const manifestPath = fs::path(fileName).parent_path();
    for (const auto& node: textures)
    {
        if (!node.isObject())
        {
            outError = "Malformed manifest: all entries in the textures array must be objects.";
            return false;
        }

        std::string const fileName = node["fileName"].asString();

        ManifestEntry entry;
        entry.fileName = (manifestPath / fileName).generic_string();
        entry.entryName = node["name"].asString();
        if (entry.entryName.empty())
            entry.entryName = fs::path(fileName).stem().generic_string();
        entry.mipLevel = node["mipLevel"].asInt();
        entry.isSRGB = node["isSRGB"].asBool();
        entry.verticalFlip = node["verticalFlip"].asBool();
        entry.channelSwizzle = node["channelSwizzle"].asString();
        auto firstChannel = node["firstChannel"];
        entry.firstChannel = firstChannel.isInt() ? firstChannel.asInt() : entry.firstChannel;

        // Normalize and validate the channel selection
        if (!entry.channelSwizzle.empty())
        {
            UppercaseString(entry.channelSwizzle);
            bool valid = true;

            if (entry.channelSwizzle.size() > 4)
                valid = false;

            for (char c : entry.channelSwizzle)
            {
                if (!strchr("RGBA", c))
                    valid = false;
            }

            if (!valid)
            {
                std::ostringstream oss;
                oss << "Invalid channel swizzle '" << entry.channelSwizzle << "' specified for texture '"
                    << entry.entryName << "'. It must be 0-4 characters long and contain only RGBA characters.";
                outError = oss.str();
                return false;
            }
        }
        
        // Parse the output format
        std::string bcFormat = node["bcFormat"].asString();
        if (bcFormat.empty())
            bcFormat = node["outputFormat"].asString(); // Legacy version
        if (!bcFormat.empty())
        {
            auto parsedFormat = ParseBlockCompressedFormat(bcFormat.c_str());
            if (parsedFormat.has_value())
            {
                entry.bcFormat = parsedFormat.value();
            }
            else
            {
                std::ostringstream oss;
                oss << "Unknown format '" << bcFormat.c_str() << "' specified for texture '" << entry.entryName << "'.";
                outError = oss.str();
                return false;
            }
        }

        // Parse the semantic bindings
        Json::Value const& semanticsNode = node["semantics"];
        if (semanticsNode.isObject())
        {
            for (std::string const& semanticName : semanticsNode.getMemberNames())
            {
                SemanticLabel label = ParseSemanticLabel(semanticName.c_str());
                if (label == SemanticLabel::None)
                {
                    std::ostringstream oss;
                    oss << "Unknown semantic label '" << semanticName.c_str() << "' specified for texture '" << entry.entryName << "'.";
                    outError = oss.str();
                    return false;
                }

                std::string channels = semanticsNode[semanticName].asString();
                UppercaseString(channels);
                static char const* channelMap = "RGBA";
                char const* firstChannelPtr = strstr(channelMap, channels.c_str());
                if (channels.empty() || !firstChannelPtr)
                {
                    std::ostringstream oss;
                    oss << "Invalid semantic binding '" << channels << "' specified for texture '" << entry.entryName
                        << "' semantic '" << semanticName << "'. Semantic bindings must use sequential channels from RGBA set.";
                    outError = oss.str();
                    return false;
                }

                int expectedChannelCount = GetSemanticChannelCount(label);
                if (int(channels.size()) != expectedChannelCount)
                {
                    std::ostringstream oss;
                    oss << "Invalid semantic binding '" << channels << "' specified for texture '" << entry.entryName
                        << "' semantic '" << semanticName << "'. This semantic requires " << expectedChannelCount << " channels.";
                    outError = oss.str();
                    return false;
                }

                ImageSemanticBinding binding;
                binding.label = label;
                binding.firstChannel = firstChannelPtr - channelMap;
                entry.semantics.push_back(binding);
            }
        }
        else if (!semanticsNode.isNull())
        {
            outError = "Malformed manifest: 'semantics' property must be an object.";
            return false;
        }

        outManifest.textures.push_back(entry);
    }

    return true;
}

void UpdateToolInputType(ToolInputType& current, ToolInputType newInput)
{
    switch(current)
    {
        case ToolInputType::None:
            // First input, use its type
            current = newInput;
            return;
        case ToolInputType::Directory:
        case ToolInputType::CompressedTextureSet:
        case ToolInputType::Manifest:
            // Mismatching input types or using more than one of these is not allowed
            current = ToolInputType::Mixed;
            return;
        case ToolInputType::Images:
            // Multiple images are allowed, mixing images with other types is not
            if (newInput != ToolInputType::Images)
                current = ToolInputType::Mixed;
            return;
    }
}
