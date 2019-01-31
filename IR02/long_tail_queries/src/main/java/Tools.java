import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class Tools {
    public static HashMap<String, Long> getMapKey1(Reducer.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, Long> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Reducer got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(keyVal[1], Long.parseLong(keyVal[0]));
        }
        br.close();
        return urlMap;
    }

    public static HashMap<String, Long> getMapKey1(Mapper.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, Long> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Mapper got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(keyVal[1], Long.parseLong(keyVal[0]));
        }
        br.close();
        return urlMap;
    }

    public static HashMap<String, Long> getMapKey0(Reducer.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, Long> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Reducer got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(keyVal[0], Long.parseLong(keyVal[1]));
        }
        br.close();
        return urlMap;
    }

    public static HashMap<String, Long> getMapKey0(Mapper.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, Long> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Mapper got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(keyVal[0], Long.parseLong(keyVal[1]));
        }
        br.close();
        return urlMap;
    }

    public static ArrayList<String> getArrayList(Reducer.Context context, Path unique_file) throws IOException, InterruptedException {
        ArrayList<String> out = new ArrayList<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        while ((str = br.readLine()) != null) {
            out.add(str);
        }
        br.close();
        return out;
    }
    public static ArrayList<String> getArrayList(Mapper.Context context, Path unique_file) throws IOException, InterruptedException {
        ArrayList<String> out = new ArrayList<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        while ((str = br.readLine()) != null) {
            out.add(str);
        }
        br.close();
        return out;
    }

    public static HashMap<Long, String> getInvMap(Reducer.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<Long, String> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Reducer got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(Long.parseLong(keyVal[0]), keyVal[1]);
        }
        br.close();
        return urlMap;
    }

    public static HashMap<Long, String> getInvMap(Mapper.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<Long, String> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Mapper got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(Long.parseLong(keyVal[0]), keyVal[1]);
        }
        br.close();
        return urlMap;
    }

    public static HashSet<String> getHashSet(Mapper.Context context, Path unique_file) throws IOException, InterruptedException {
        HashSet<String> out = new HashSet<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        while ((str = br.readLine()) != null) {
            out.add(str);
        }
        br.close();
        return out;
    }

    public static HashSet<String> getHashSet(Reducer.Context context, Path unique_file) throws IOException, InterruptedException {
        HashSet<String> out = new HashSet<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        while ((str = br.readLine()) != null) {
            out.add(str);
        }
        br.close();
        return out;
    }

    public static HashMap<String, String> getSMap(Reducer.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, String> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Reducer got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(keyVal[0], keyVal[1]);
        }
        br.close();
        return urlMap;
    }

    public static HashMap<String, String> getSMap(Mapper.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, String> urlMap = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split(sep);
            if (verbose) {
                System.out.println("Reducer got key\t::" + keyVal[1] + "::");
            }
            urlMap.put(keyVal[0], keyVal[1]);
        }
        br.close();
        return urlMap;
    }

    public static HashMap<String, HashSet<String>> getHashSetMap(Mapper.Context context, Path unique_file, boolean verbose, String sep) throws IOException, InterruptedException {
        HashMap<String, HashSet<String>> res = new HashMap<>();
        FileSystem fs = unique_file.getFileSystem(context.getConfiguration());
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(unique_file), StandardCharsets.UTF_8));
        String str;
        String[] keyVal;
        String[] setItems;
        while ((str = br.readLine()) != null) {
            keyVal = str.trim().split("\t");
            setItems = keyVal[1].split(sep);
            HashSet<String> hashSetItems = new HashSet<>();
            for (String i: setItems){
                hashSetItems.add(i);
            }
            res.put(keyVal[0], hashSetItems);
            if (verbose) {
                System.out.println("Reducer got key\t::" + keyVal[0] + "::");
            }
        }
        br.close();
        return res;
    }


}
